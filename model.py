# model.py
"""
PI-EBM: Physics-Informed Energy-Based Model for Earthquake Prediction
======================================================================
Version 2.0 - Optimized for F1 > 0.55

Key changes from v1:
- Local context features (seismicity history around event)
- Two-stage training (predictions first, then physics)
- Weighted sampling for class imbalance
- Simplified energy integration
- Label smoothing for calibration
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the PI-EBM model"""
    
    # Spatial grid settings
    lat_bins: int = 90
    lon_bins: int = 180
    
    # Temporal settings
    sequence_length: int = 30
    prediction_horizon: int = 7
    
    # Multi-scale temporal windows (days)
    temporal_scales: Tuple[int, ...] = (7, 30, 90)
    
    # Model architecture
    hidden_dim: int = 192
    latent_dim: int = 64
    num_attention_heads: int = 4
    dropout: float = 0.25  # Reduced from 0.3
    
    # Energy-based model settings
    ebm_hidden_dim: int = 128
    energy_reg_weight: float = 0.005  # Reduced
    contrastive_margin: float = 0.5  # Reduced
    
    # Physics constraints - REDUCED for stage 1
    lambda_physics: float = 0.1  # Was 0.5
    lambda_contrastive: float = 0.1  # Was 0.2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80  # More epochs
    patience: int = 12  # More patience
    warmup_epochs: int = 5
    
    # Label smoothing
    label_smoothing: float = 0.05
    
    # Focal loss for imbalanced classes
    focal_gamma: float = 2.0
    
    # Class weights
    tsunami_pos_weight: float = 50.0
    
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Data settings
    min_magnitude_trigger: float = 5.0
    aftershock_magnitude_threshold: float = 3.0
    
    # NEW: Feature settings
    event_feature_dim: int = 16  # Increased from 12


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = -target * torch.log(pred) * self.pos_weight - (1 - target) * torch.log(1 - pred)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class LabelSmoothingBCE(nn.Module):
    """Binary cross entropy with label smoothing"""
    
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred.clamp(1e-7, 1-1e-7), target_smooth)


# =============================================================================
# DATA PROCESSING
# =============================================================================

class EarthquakeDataProcessor:
    """Processes raw earthquake catalogs into model-ready tensors."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.lat_edges = np.linspace(-90, 90, config.lat_bins + 1)
        self.lon_edges = np.linspace(-180, 180, config.lon_bins + 1)
        self.stats = {}
        self._df_cache = None  # Cache for feature extraction
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess earthquake CSV"""
        print(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        df.columns = df.columns.str.lower().str.strip()
        
        time_col = 'time' if 'time' in df.columns else 'date'
        df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        
        required_cols = ['datetime', 'latitude', 'longitude', 'magnitude']
        df = df.dropna(subset=required_cols)
        
        df = df[
            (df['magnitude'] >= 0) &
            (df['latitude'].between(-90, 90)) &
            (df['longitude'].between(-180, 180))
        ]
        
        df = df.sort_values('datetime').reset_index(drop=True)
        
        df['date'] = df['datetime'].dt.date
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['dayofyear'] = df['datetime'].dt.dayofyear
        df['hour'] = df['datetime'].dt.hour
        
        if 'depth' not in df.columns:
            df['depth'] = 10.0
        df['depth'] = df['depth'].fillna(10.0).clip(0, 700)
        
        if 'tsunami' not in df.columns:
            df['tsunami'] = 0
        df['tsunami'] = df['tsunami'].fillna(0).astype(int)
        
        self.stats = {
            'mag_mean': df['magnitude'].mean(),
            'mag_std': df['magnitude'].std(),
            'depth_mean': df['depth'].mean(),
            'depth_std': df['depth'].std(),
        }
        
        self._df_cache = df
        
        print(f"Loaded {len(df):,} earthquakes ({df['year'].min()}-{df['year'].max()})")
        print(f"Magnitude range: {df['magnitude'].min():.1f} - {df['magnitude'].max():.1f}")
        
        return df
    
    def create_spatiotemporal_grid(
        self, 
        df: pd.DataFrame, 
        end_date: pd.Timestamp,
        lookback_days: int = None
    ) -> np.ndarray:
        """Create a multi-channel spatial grid summarizing seismic activity."""
        if lookback_days is None:
            lookback_days = self.config.sequence_length
        
        start_date = end_date - pd.Timedelta(days=lookback_days)
        mask = (df['datetime'] >= start_date) & (df['datetime'] < end_date)
        window_df = df[mask]
        
        grid = np.zeros((6, self.config.lat_bins, self.config.lon_bins), dtype=np.float32)
        
        if len(window_df) == 0:
            return grid
        
        lat_idx = np.digitize(window_df['latitude'].values, self.lat_edges) - 1
        lon_idx = np.digitize(window_df['longitude'].values, self.lon_edges) - 1
        
        lat_idx = np.clip(lat_idx, 0, self.config.lat_bins - 1)
        lon_idx = np.clip(lon_idx, 0, self.config.lon_bins - 1)
        
        magnitudes = window_df['magnitude'].values
        depths = window_df['depth'].values
        
        mag_sum = np.zeros((self.config.lat_bins, self.config.lon_bins), dtype=np.float32)
        mag_sq_sum = np.zeros((self.config.lat_bins, self.config.lon_bins), dtype=np.float32)
        
        for i in range(len(window_df)):
            li, lo = lat_idx[i], lon_idx[i]
            grid[0, li, lo] += 1
            grid[1, li, lo] = max(grid[1, li, lo], magnitudes[i])
            grid[2, li, lo] += 10 ** (1.5 * magnitudes[i] + 4.8)
            grid[3, li, lo] += depths[i]
            mag_sum[li, lo] += magnitudes[i]
            mag_sq_sum[li, lo] += magnitudes[i] ** 2
        
        nonzero_mask = grid[0] > 0
        grid[3, nonzero_mask] /= grid[0, nonzero_mask]
        
        mean_mag = np.zeros_like(mag_sum)
        mean_mag[nonzero_mask] = mag_sum[nonzero_mask] / grid[0, nonzero_mask]
        variance = np.zeros_like(mag_sum)
        variance[nonzero_mask] = (mag_sq_sum[nonzero_mask] / grid[0, nonzero_mask]) - mean_mag[nonzero_mask] ** 2
        grid[5] = np.sqrt(np.maximum(variance, 0)) / 3.0
        
        grid[0] = np.log1p(grid[0])
        grid[1] = grid[1] / 10.0
        grid[2] = np.log10(grid[2] + 1) / 20.0
        grid[3] = grid[3] / 700.0
        
        mid_date = end_date - pd.Timedelta(days=lookback_days // 2)
        recent_mask = (df['datetime'] >= mid_date) & (df['datetime'] < end_date)
        older_mask = (df['datetime'] >= start_date) & (df['datetime'] < mid_date)
        
        recent_count = len(df[recent_mask])
        older_count = len(df[older_mask])
        
        if older_count > 0:
            trend_ratio = recent_count / (older_count + 1)
            grid[4] = grid[0] * np.clip(trend_ratio, 0.1, 10) / 5.0
        else:
            grid[4] = grid[0]
        
        return grid
    
    def create_multiscale_grids(self, df: pd.DataFrame, end_date: pd.Timestamp) -> np.ndarray:
        """Create grids at multiple temporal scales."""
        grids = []
        for scale in self.config.temporal_scales:
            grid = self.create_spatiotemporal_grid(df, end_date, lookback_days=scale)
            grids.append(grid)
        return np.concatenate(grids, axis=0)
    
    def extract_event_features(self, row: pd.Series, df: pd.DataFrame = None) -> np.ndarray:
        """
        Extract normalized features with LOCAL CONTEXT.
        This is the key improvement - we add seismicity history around the event.
        """
        if df is None:
            df = self._df_cache
        
        depth = row.get('depth', 10.0)
        if pd.isna(depth):
            depth = 10.0
        
        mag = row['magnitude']
        lat = row['latitude']
        lon = row['longitude']
        event_time = row['datetime']
        
        # === LOCAL SEISMICITY FEATURES (KEY IMPROVEMENT) ===
        local_count_7d = 0
        local_count_30d = 0
        local_max_mag = 0
        local_energy = 0
        mag_deficit = 0
        recent_trend = 0
        
        if df is not None:
            # 7-day local window
            start_7d = event_time - pd.Timedelta(days=7)
            start_30d = event_time - pd.Timedelta(days=30)
            
            # Adaptive radius based on latitude
            lat_range = 2.0
            lon_range = 2.0 / max(np.cos(np.radians(lat)), 0.1)
            
            local_mask_30d = (
                (df['datetime'] >= start_30d) &
                (df['datetime'] < event_time) &
                (df['latitude'].between(lat - lat_range, lat + lat_range)) &
                (df['longitude'].between(lon - lon_range, lon + lon_range))
            )
            local_df_30d = df[local_mask_30d]
            
            local_mask_7d = (
                (df['datetime'] >= start_7d) &
                (df['datetime'] < event_time) &
                (df['latitude'].between(lat - lat_range, lat + lat_range)) &
                (df['longitude'].between(lon - lon_range, lon + lon_range))
            )
            local_df_7d = df[local_mask_7d]
            
            local_count_7d = len(local_df_7d)
            local_count_30d = len(local_df_30d)
            
            if local_count_30d > 0:
                local_max_mag = local_df_30d['magnitude'].max()
                local_energy = np.log10(np.sum(10 ** (1.5 * local_df_30d['magnitude'].values)) + 1)
                mag_deficit = mag - local_max_mag
            
            # Trend: recent vs older activity
            if local_count_30d > local_count_7d and (local_count_30d - local_count_7d) > 0:
                recent_trend = local_count_7d / (local_count_30d - local_count_7d + 1)
            else:
                recent_trend = 1.0
        
        # Depth categories
        is_shallow = float(depth < 70)
        is_intermediate = float(70 <= depth < 300)
        is_deep = float(depth >= 300)
        
        # Build feature vector (16 features)
        features = np.array([
            # Basic event features
            mag / 10.0,
            (lat + 90) / 180.0,
            (lon + 180) / 360.0,
            depth / 700.0,
            # Temporal features
            np.sin(2 * np.pi * row['dayofyear'] / 365.0),
            np.cos(2 * np.pi * row['dayofyear'] / 365.0),
            row.get('hour', 12) / 24.0,
            # Depth categories
            is_shallow,
            is_intermediate,
            is_deep,
            # LOCAL CONTEXT (NEW - key for performance)
            np.log1p(local_count_7d) / 5.0,
            np.log1p(local_count_30d) / 6.0,
            local_max_mag / 10.0,
            local_energy / 15.0,
            np.clip(mag_deficit + 2, 0, 4) / 4.0,  # Is this larger than recent events?
            np.clip(recent_trend, 0, 5) / 5.0,  # Activity trend
        ], dtype=np.float32)
        
        return features
    
    def compute_physics_features(
        self,
        df: pd.DataFrame,
        event_row: pd.Series,
        lookback_days: int = 30
    ) -> Dict[str, np.ndarray]:
        """Compute features for physics loss computation."""
        event_time = event_row['datetime']
        event_lat = event_row['latitude']
        event_lon = event_row['longitude']
        
        start_time = event_time - pd.Timedelta(days=lookback_days)
        
        lat_range = 2.0
        lon_range = 2.0 / max(np.cos(np.radians(event_lat)), 0.1)
        
        local_mask = (
            (df['datetime'] >= start_time) &
            (df['datetime'] < event_time) &
            (df['latitude'].between(event_lat - lat_range, event_lat + lat_range)) &
            (df['longitude'].between(event_lon - lon_range, event_lon + lon_range))
        )
        local_df = df[local_mask]
        
        mag_bins = np.arange(0, 10, 0.5)
        mag_counts, _ = np.histogram(local_df['magnitude'].values, bins=mag_bins)
        mag_counts = mag_counts.astype(np.float32) + 1e-6
        
        if len(local_df) > 0:
            time_diffs = (event_time - local_df['datetime']).dt.total_seconds() / 86400.0
            time_bins = np.array([0, 1, 2, 3, 5, 7, 14, 21, 30], dtype=np.float32)
            time_counts, _ = np.histogram(time_diffs.values, bins=time_bins)
            time_counts = time_counts.astype(np.float32) + 1e-6
        else:
            time_counts = np.ones(8, dtype=np.float32) * 1e-6
        
        return {
            'magnitude_counts': mag_counts,
            'time_counts': time_counts,
            'time_bins': np.array([0.5, 1.5, 2.5, 4, 6, 10.5, 17.5, 25.5], dtype=np.float32),
        }
    
    def compute_labels(self, df: pd.DataFrame, event_row: pd.Series) -> Dict[str, float]:
        """Compute multi-task labels for a mainshock event."""
        event_time = event_row['datetime']
        event_lat = event_row['latitude']
        event_lon = event_row['longitude']
        event_mag = event_row['magnitude']
        
        start = event_time
        end = event_time + pd.Timedelta(days=self.config.prediction_horizon)
        
        lat_range = 1.0
        lon_range = 1.0 / max(np.cos(np.radians(event_lat)), 0.1)
        
        aftershock_mask = (
            (df['datetime'] > start) & 
            (df['datetime'] <= end) &
            (df['latitude'].between(event_lat - lat_range, event_lat + lat_range)) &
            (df['longitude'].between(event_lon - lon_range, event_lon + lon_range)) &
            (df['magnitude'] >= self.config.aftershock_magnitude_threshold)
        )
        aftershocks = df[aftershock_mask]
        
        larger_event_mask = aftershock_mask & (df['magnitude'] > event_mag)
        is_foreshock = larger_event_mask.any()
        
        aftershock_count = len(aftershocks)
        max_aftershock_mag = aftershocks['magnitude'].max() if aftershock_count > 0 else 0.0
        
        return {
            'aftershock_count': min(aftershock_count, 100) / 100.0,
            'max_aftershock_mag': max_aftershock_mag / 10.0,
            'has_aftershocks': float(aftershock_count > 0),
            'tsunami': float(event_row.get('tsunami', 0)),
            'is_foreshock': float(is_foreshock),
        }
    
    def create_dataset(
        self, 
        df: pd.DataFrame,
        min_magnitude: float = None,
        show_progress: bool = True
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], List[Dict]]:
        """Create complete dataset from earthquake catalog."""
        if min_magnitude is None:
            min_magnitude = self.config.min_magnitude_trigger
        
        self._df_cache = df
        
        significant = df[df['magnitude'] >= min_magnitude].copy()
        
        min_date = df['datetime'].min() + pd.Timedelta(days=max(self.config.temporal_scales) + 7)
        max_date = df['datetime'].max() - pd.Timedelta(days=self.config.prediction_horizon)
        
        significant = significant[
            (significant['datetime'] >= min_date) &
            (significant['datetime'] <= max_date)
        ]
        
        print(f"Creating dataset from {len(significant):,} M{min_magnitude}+ events...")
        
        grids = []
        event_features = []
        labels = []
        physics_features = []
        
        iterator = tqdm(significant.iterrows(), total=len(significant)) if show_progress else significant.iterrows()
        
        for idx, row in iterator:
            try:
                context_date = row['datetime'] - pd.Timedelta(hours=1)
                grid = self.create_multiscale_grids(df, context_date)
                feat = self.extract_event_features(row, df)  # Pass df for local context
                label = self.compute_labels(df, row)
                physics = self.compute_physics_features(df, row)
                
                grids.append(grid)
                event_features.append(feat)
                labels.append(label)
                physics_features.append(physics)
                
            except Exception as e:
                continue
        
        print(f"Created {len(grids):,} training samples")
        
        return grids, event_features, labels, physics_features


# =============================================================================
# DATASET
# =============================================================================

class EarthquakeDataset(Dataset):
    """PyTorch Dataset for earthquake sequences"""
    
    def __init__(
        self, 
        grids: List[np.ndarray],
        event_features: List[np.ndarray],
        labels: List[Dict],
        physics_features: List[Dict]
    ):
        self.grids = [torch.FloatTensor(g) for g in grids]
        self.event_features = [torch.FloatTensor(f) for f in event_features]
        self.labels = labels
        self.physics_features = physics_features
    
    def __len__(self) -> int:
        return len(self.grids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label_dict = self.labels[idx]
        physics = self.physics_features[idx]
        
        label_tensor = torch.FloatTensor([
            label_dict['aftershock_count'],
            label_dict['max_aftershock_mag'],
            label_dict['has_aftershocks'],
            label_dict['tsunami'],
            label_dict['is_foreshock'],
        ])
        
        return {
            'grid': self.grids[idx],
            'event_features': self.event_features[idx],
            'labels': label_tensor,
            'magnitude_counts': torch.FloatTensor(physics['magnitude_counts']),
            'time_counts': torch.FloatTensor(physics['time_counts']),
            'time_bins': torch.FloatTensor(physics['time_bins']),
        }


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class EfficientChannelAttention(nn.Module):
    """Lightweight channel attention"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Lightweight spatial attention"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class ConvBlock(nn.Module):
    """Convolutional block with attention"""
    
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        self.channel_attn = EfficientChannelAttention(out_channels) if use_attention else nn.Identity()
        self.spatial_attn = SpatialAttention() if use_attention else nn.Identity()
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x + residual


class SpatiotemporalEncoder(nn.Module):
    """Multi-scale CNN encoder for earthquake activity grids."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        n_scales = len(config.temporal_scales)
        in_channels = 6 * n_scales
        
        self.conv1 = ConvBlock(in_channels, 48, use_attention=True, dropout=0.1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(48, 96, use_attention=True, dropout=0.1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(96, 144, use_attention=True, dropout=0.15)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = ConvBlock(144, 192, use_attention=False, dropout=0.15)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        x = self.global_pool(x)
        return self.fc(x)


class EventEncoder(nn.Module):
    """MLP encoder for earthquake features with local context"""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, output_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnergyFunction(nn.Module):
    """
    Simplified Energy-based scoring network.
    Key change: concatenate energy instead of gating.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Simpler activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
    
    def modulate(self, z: torch.Tensor) -> torch.Tensor:
        """Concatenate energy as additional feature instead of gating"""
        energy = self.net(z)  # (batch, 1)
        energy_norm = torch.tanh(energy)  # Normalize to [-1, 1]
        return torch.cat([z, energy_norm], dim=-1)


class PhysicsConstraints(nn.Module):
    """Physics-informed regularization module."""
    
    def __init__(self):
        super().__init__()
        
        self.log_b_value = nn.Parameter(torch.tensor(0.0))
        self.log_p_value = nn.Parameter(torch.tensor(0.0))
        self.log_c_value = nn.Parameter(torch.tensor(-4.6))
        self.delta_m = nn.Parameter(torch.tensor(1.2))
    
    @property
    def b_value(self) -> torch.Tensor:
        return 0.7 + 0.6 * torch.sigmoid(self.log_b_value)
    
    @property
    def p_value(self) -> torch.Tensor:
        return 0.8 + 0.4 * torch.sigmoid(self.log_p_value)
    
    @property
    def c_value(self) -> torch.Tensor:
        return F.softplus(self.log_c_value) + 0.001
    
    def gutenberg_richter_loss(self, magnitude_counts: torch.Tensor) -> torch.Tensor:
        batch_size, n_bins = magnitude_counts.shape
        log_counts = torch.log10(magnitude_counts + 1)
        mags = torch.arange(n_bins, device=magnitude_counts.device, dtype=torch.float32) * 0.5
        mags = mags.unsqueeze(0).expand(batch_size, -1)
        a_value = log_counts[:, 0:1]
        expected = a_value - self.b_value * mags
        weight = (magnitude_counts > 0.5).float()
        loss = (weight * (log_counts - expected) ** 2).sum(dim=1) / (weight.sum(dim=1) + 1e-8)
        return loss.mean()
    
    def omori_loss(self, time_bins: torch.Tensor, time_counts: torch.Tensor) -> torch.Tensor:
        predicted = 1.0 / (time_bins + self.c_value) ** self.p_value
        predicted = predicted / (predicted.sum(dim=1, keepdim=True) + 1e-8)
        actual = time_counts / (time_counts.sum(dim=1, keepdim=True) + 1e-8)
        loss = F.kl_div(torch.log(predicted + 1e-8), actual, reduction='batchmean')
        return loss
    
    def bath_law_loss(self, mainshock_mag: torch.Tensor, max_aftershock_mag: torch.Tensor) -> torch.Tensor:
        expected_diff = self.delta_m
        actual_diff = mainshock_mag - max_aftershock_mag
        valid_mask = max_aftershock_mag > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=mainshock_mag.device)
        loss = F.mse_loss(actual_diff[valid_mask], expected_diff.expand_as(actual_diff[valid_mask]))
        return loss
    
    def get_params(self) -> Dict[str, float]:
        return {
            'b_value': self.b_value.item(),
            'p_value': self.p_value.item(),
            'c_value': self.c_value.item(),
            'delta_m': self.delta_m.item(),
        }


class PredictionHeads(nn.Module):
    """Multi-task prediction heads"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.25):
        super().__init__()
        
        # +1 for energy feature from concatenation
        actual_input = input_dim + 1
        
        self.shared = nn.Sequential(
            nn.Linear(actual_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.aftershock_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )
        
        self.tsunami_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.foreshock_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared(z)
        
        aftershock_out = self.aftershock_head(shared)
        tsunami_out = self.tsunami_head(shared)
        foreshock_out = self.foreshock_head(shared)
        
        return {
            'aftershock_count': torch.sigmoid(aftershock_out[:, 0]),
            'max_aftershock_mag': torch.sigmoid(aftershock_out[:, 1]),
            'has_aftershocks': torch.sigmoid(aftershock_out[:, 2]),
            'tsunami': torch.sigmoid(tsunami_out.squeeze(-1)),
            'is_foreshock': torch.sigmoid(foreshock_out.squeeze(-1)),
        }


# =============================================================================
# MAIN MODEL
# =============================================================================

class PIEBM(nn.Module):
    """
    Physics-Informed Energy-Based Model for Earthquake Prediction
    Version 2.0 - Optimized
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.spatial_encoder = SpatiotemporalEncoder(config)
        self.event_encoder = EventEncoder(
            input_dim=config.event_feature_dim,  # Now 16
            hidden_dim=64,
            output_dim=64,
            dropout=config.dropout
        )
        
        fusion_input_dim = config.hidden_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.latent_dim * 2),
            nn.LayerNorm(config.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
        )
        
        self.energy_fn = EnergyFunction(
            config.latent_dim,
            config.ebm_hidden_dim,
            dropout=config.dropout
        )
        
        self.prediction_heads = PredictionHeads(
            config.latent_dim,  # Will be +1 inside for energy
            hidden_dim=config.hidden_dim // 2,
            dropout=config.dropout
        )
        
        self.physics = PhysicsConstraints()
    
    def encode(self, grid: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_encoder(grid)
        event_emb = self.event_encoder(event_features)
        combined = torch.cat([spatial_features, event_emb], dim=1)
        z = self.fusion(combined)
        return z
    
    def forward(self, grid: torch.Tensor, event_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(grid, event_features)
        
        # Concatenate energy as feature
        z_with_energy = self.energy_fn.modulate(z)
        
        predictions = self.prediction_heads(z_with_energy)
        predictions['latent'] = z
        predictions['energy'] = self.energy_fn(z)
        
        return predictions
    
    def compute_physics_loss(
        self,
        magnitude_counts: torch.Tensor,
        time_bins: torch.Tensor,
        time_counts: torch.Tensor,
        mainshock_mag: torch.Tensor,
        max_aftershock_mag: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        gr_loss = self.physics.gutenberg_richter_loss(magnitude_counts)
        omori_loss = self.physics.omori_loss(time_bins, time_counts)
        bath_loss = self.physics.bath_law_loss(mainshock_mag, max_aftershock_mag)
        
        return {
            'gutenberg_richter': gr_loss,
            'omori': omori_loss,
            'bath': bath_loss,
            'physics_total': gr_loss + omori_loss + 0.5 * bath_loss,
        }
    
    @torch.no_grad()
    def compute_anomaly_score(self, grid: torch.Tensor, event_features: torch.Tensor) -> torch.Tensor:
        z = self.encode(grid, event_features)
        return self.energy_fn(z)


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """Two-stage training with physics losses and class balancing"""
    
    def __init__(self, model: PIEBM, config: ModelConfig):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.focal_loss = FocalLoss(gamma=config.focal_gamma, pos_weight=config.tsunami_pos_weight)
        self.smooth_bce = LabelSmoothingBCE(smoothing=config.label_smoothing)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
        }
        
        # Stage tracking
        self.current_stage = 1
        self.stage1_lambda_physics = 0.0
        self.stage2_lambda_physics = config.lambda_physics
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components"""
        losses = {}
        
        # Regression losses
        losses['aftershock_count'] = F.mse_loss(predictions['aftershock_count'], labels[:, 0])
        losses['max_aftershock_mag'] = F.huber_loss(predictions['max_aftershock_mag'], labels[:, 1], delta=0.1)
        
        # Classification losses with label smoothing
        losses['has_aftershocks'] = self.smooth_bce(predictions['has_aftershocks'], labels[:, 2])
        
        # Focal loss for tsunami
        losses['tsunami'] = self.focal_loss(predictions['tsunami'], labels[:, 3])
        
        # Foreshock with class weighting
        pos_weight = 3.0
        weight = torch.where(labels[:, 4] > 0.5, pos_weight, 1.0)
        losses['is_foreshock'] = (F.binary_cross_entropy(
            predictions['is_foreshock'].clamp(1e-7, 1-1e-7),
            labels[:, 4],
            reduction='none'
        ) * weight).mean()
        
        # Contrastive energy loss
        z = predictions['latent']
        e_real = predictions['energy']
        z_neg = z + torch.randn_like(z) * 0.3
        e_neg = self.model.energy_fn(z_neg)
        losses['contrastive'] = F.softplus(e_real - e_neg + self.config.contrastive_margin).mean()
        
        # Energy regularization
        losses['energy_reg'] = self.config.energy_reg_weight * (e_real ** 2).mean()
        
        # Physics losses - only in stage 2
        current_lambda = self.stage1_lambda_physics if self.current_stage == 1 else self.stage2_lambda_physics
        
        if current_lambda > 0:
            mainshock_mag = batch['event_features'][:, 0] * 10
            max_aftershock_mag = labels[:, 1] * 10
            
            physics_losses = self.model.compute_physics_loss(
                batch['magnitude_counts'],
                batch['time_bins'],
                batch['time_counts'],
                mainshock_mag,
                max_aftershock_mag
            )
            losses.update(physics_losses)
        else:
            losses['physics_total'] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total = (
            losses['aftershock_count'] * 1.0 +
            losses['max_aftershock_mag'] * 1.0 +
            losses['has_aftershocks'] * 1.5 +  # Slightly higher weight
            losses['tsunami'] * 1.0 +
            losses['is_foreshock'] * 1.0 +
            losses['contrastive'] * self.config.lambda_contrastive +
            losses['energy_reg'] +
            losses['physics_total'] * current_lambda
        )
        
        losses['total'] = total
        return losses
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            grid = batch['grid'].to(self.device)
            event_features = batch['event_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            batch_device = {
                'event_features': event_features,
                'magnitude_counts': batch['magnitude_counts'].to(self.device),
                'time_counts': batch['time_counts'].to(self.device),
                'time_bins': batch['time_bins'].to(self.device),
            }
            
            predictions = self.model(grid, event_features)
            losses = self.compute_loss(predictions, labels, batch_device)
            
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        all_preds = {k: [] for k in ['aftershock_count', 'has_aftershocks', 'has_aftershocks_prob', 
                                       'tsunami', 'tsunami_prob', 'is_foreshock', 'is_foreshock_prob', 'energy']}
        all_labels = {k: [] for k in ['aftershock_count', 'has_aftershocks', 'tsunami', 'is_foreshock']}
        
        total_loss = 0.0
        
        for batch in dataloader:
            grid = batch['grid'].to(self.device)
            event_features = batch['event_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            batch_device = {
                'event_features': event_features,
                'magnitude_counts': batch['magnitude_counts'].to(self.device),
                'time_counts': batch['time_counts'].to(self.device),
                'time_bins': batch['time_bins'].to(self.device),
            }
            
            predictions = self.model(grid, event_features)
            losses = self.compute_loss(predictions, labels, batch_device)
            total_loss += losses['total'].item()
            
            all_preds['aftershock_count'].extend(predictions['aftershock_count'].cpu().numpy())
            all_preds['has_aftershocks_prob'].extend(predictions['has_aftershocks'].cpu().numpy())
            all_preds['has_aftershocks'].extend((predictions['has_aftershocks'] > 0.5).float().cpu().numpy())
            all_preds['tsunami_prob'].extend(predictions['tsunami'].cpu().numpy())
            all_preds['tsunami'].extend((predictions['tsunami'] > 0.3).float().cpu().numpy())
            all_preds['is_foreshock_prob'].extend(predictions['is_foreshock'].cpu().numpy())
            all_preds['is_foreshock'].extend((predictions['is_foreshock'] > 0.5).float().cpu().numpy())
            all_preds['energy'].extend(predictions['energy'].cpu().numpy())
            
            all_labels['aftershock_count'].extend(labels[:, 0].cpu().numpy())
            all_labels['has_aftershocks'].extend(labels[:, 2].cpu().numpy())
            all_labels['tsunami'].extend(labels[:, 3].cpu().numpy())
            all_labels['is_foreshock'].extend(labels[:, 4].cpu().numpy())
        
        metrics = {'val_loss': total_loss / len(dataloader)}
        
        metrics['aftershock_count_mse'] = np.mean(
            (np.array(all_preds['aftershock_count']) - np.array(all_labels['aftershock_count'])) ** 2
        )
        
        for task in ['has_aftershocks', 'tsunami', 'is_foreshock']:
            preds = np.array(all_preds[task])
            probs = np.array(all_preds[f'{task}_prob'])
            labels_arr = np.array(all_labels[task])
            
            metrics[f'{task}_accuracy'] = np.mean(preds == labels_arr)
            
            tp = np.sum((preds == 1) & (labels_arr == 1))
            fp = np.sum((preds == 1) & (labels_arr == 0))
            fn = np.sum((preds == 0) & (labels_arr == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics[f'{task}_precision'] = precision
            metrics[f'{task}_recall'] = recall
            metrics[f'{task}_f1'] = f1
            
            if len(np.unique(labels_arr)) > 1:
                from sklearn.metrics import roc_auc_score
                try:
                    metrics[f'{task}_auc'] = roc_auc_score(labels_arr, probs)
                except:
                    metrics[f'{task}_auc'] = 0.5
            else:
                metrics[f'{task}_auc'] = 0.5
        
        energies = np.array(all_preds['energy'])
        metrics['energy_mean'] = np.mean(energies)
        metrics['energy_std'] = np.std(energies)
        
        physics_params = self.model.physics.get_params()
        metrics.update({f'physics_{k}': v for k, v in physics_params.items()})
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = None,
        patience: int = None
    ):
        """Two-stage training: predictions first, then physics"""
        
        if epochs is None:
            epochs = self.config.epochs
        if patience is None:
            patience = self.config.patience
        
        # Stage 1: Train predictions only (no physics)
        stage1_epochs = epochs // 3
        print(f"\n{'='*60}")
        print(f"STAGE 1: Prediction Training ({stage1_epochs} epochs, no physics)")
        print(f"{'='*60}")
        
        self.current_stage = 1
        best_f1 = 0.0
        best_model_state = None
        no_improve = 0
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=stage1_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        for epoch in range(stage1_epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            if val_loader:
                metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(metrics['val_loss'])
                self.history['metrics'].append(metrics)
                
                combined_f1 = (
                    metrics['has_aftershocks_f1'] * 0.5 +
                    metrics['tsunami_f1'] * 0.3 +
                    metrics['is_foreshock_f1'] * 0.2
                )
                
                if combined_f1 > best_f1:
                    best_f1 = combined_f1
                    no_improve = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    no_improve += 1
                
                if epoch % 5 == 0 or epoch == stage1_epochs - 1:
                    print(f"Epoch {epoch:3d}/{stage1_epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"AS-F1: {metrics['has_aftershocks_f1']:.3f} | "
                          f"TS-F1: {metrics['tsunami_f1']:.3f} | "
                          f"FS-F1: {metrics['is_foreshock_f1']:.3f} | "
                          f"Avg: {combined_f1:.3f}")
        
        print(f"\nStage 1 complete. Best F1: {best_f1:.4f}")
        
        # Stage 2: Fine-tune with physics
        stage2_epochs = epochs - stage1_epochs
        print(f"\n{'='*60}")
        print(f"STAGE 2: Physics Fine-tuning ({stage2_epochs} epochs)")
        print(f"{'='*60}")
        
        self.current_stage = 2
        
        # Lower learning rate for stage 2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.learning_rate * 0.3
        
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=stage2_epochs,
            eta_min=1e-6
        )
        
        no_improve = 0
        
        for epoch in range(stage2_epochs):
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            scheduler2.step()
            
            if val_loader:
                metrics = self.evaluate(val_loader)
                self.history['val_loss'].append(metrics['val_loss'])
                self.history['metrics'].append(metrics)
                
                combined_f1 = (
                    metrics['has_aftershocks_f1'] * 0.5 +
                    metrics['tsunami_f1'] * 0.3 +
                    metrics['is_foreshock_f1'] * 0.2
                )
                
                if combined_f1 > best_f1:
                    best_f1 = combined_f1
                    no_improve = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    no_improve += 1
                
                if epoch % 5 == 0 or epoch == stage2_epochs - 1:
                    physics = self.model.physics.get_params()
                    print(f"Epoch {epoch:3d}/{stage2_epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"AS-F1: {metrics['has_aftershocks_f1']:.3f} | "
                          f"TS-F1: {metrics['tsunami_f1']:.3f} | "
                          f"FS-F1: {metrics['is_foreshock_f1']:.3f} | "
                          f"b={physics['b_value']:.2f}")
                
                if no_improve >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✓ Restored best model (combined F1: {best_f1:.4f})")
        
        print(f"\n{'='*60}")
        print(f"Training complete. Best combined F1: {best_f1:.4f}")
        print(f"{'='*60}")
        
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline"""
    
    print("=" * 70)
    print("PI-EBM v2.0: Optimized for F1 > 0.55")
    print("=" * 70)
    
    config = ModelConfig()
    print(f"\nDevice: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs} (patience={config.patience})")
    print(f"Event features: {config.event_feature_dim} (including local context)")
    
    print("\n[1/4] Loading earthquake data...")
    processor = EarthquakeDataProcessor(config)
    df = processor.load_data('earthquake_dataset/earthquakes_clean.csv')
    
    print("\n[2/4] Creating training dataset...")
    grids, event_features, labels, physics_features = processor.create_dataset(df)
    
    if len(grids) < 50:
        print("Dataset too small. Lowering magnitude threshold...")
        grids, event_features, labels, physics_features = processor.create_dataset(df, min_magnitude=4.0)
    
    dataset = EarthquakeDataset(grids, event_features, labels, physics_features)
    
    # Compute class weights
    tsunami_labels = [labels[i]['tsunami'] for i in range(len(labels))]
    aftershock_labels = [labels[i]['has_aftershocks'] for i in range(len(labels))]
    foreshock_labels = [labels[i]['is_foreshock'] for i in range(len(labels))]
    
    tsunami_ratio = sum(tsunami_labels) / len(tsunami_labels)
    if tsunami_ratio > 0:
        config.tsunami_pos_weight = min((1 - tsunami_ratio) / tsunami_ratio, 100.0)
    print(f"   Tsunami positive weight: {config.tsunami_pos_weight:.1f}")
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Weighted sampling for training
    train_indices = train_dataset.indices
    sample_weights = []
    for idx in train_indices:
        w = 1.0
        if labels[idx]['tsunami'] == 1:
            w += 10.0  # Heavily weight tsunami
        if labels[idx]['is_foreshock'] == 1:
            w += 3.0  # Weight foreshocks
        if labels[idx]['has_aftershocks'] == 0:
            w += 0.5  # Slightly weight non-aftershock
        sample_weights.append(w)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        sampler=sampler,
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=0
    )
    
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    
    print(f"\n   Class distribution:")
    print(f"   - Has aftershocks: {sum(aftershock_labels)/len(aftershock_labels)*100:.1f}%")
    print(f"   - Tsunami: {sum(tsunami_labels)/len(tsunami_labels)*100:.2f}%")
    print(f"   - Is foreshock: {sum(foreshock_labels)/len(foreshock_labels)*100:.1f}%")
    
    print("\n[3/4] Initializing PI-EBM model...")
    model = PIEBM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print("\n[4/4] Training model...")
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    final_metrics = trainer.evaluate(val_loader)
    
    # Calculate average F1
    avg_f1 = (
        final_metrics['has_aftershocks_f1'] * 0.5 +
        final_metrics['tsunami_f1'] * 0.3 +
        final_metrics['is_foreshock_f1'] * 0.2
    )
    
    print("\nPrediction Performance:")
    print(f"   Aftershock Detection:  Acc={final_metrics['has_aftershocks_accuracy']:.3f}, "
          f"P={final_metrics['has_aftershocks_precision']:.3f}, "
          f"R={final_metrics['has_aftershocks_recall']:.3f}, "
          f"F1={final_metrics['has_aftershocks_f1']:.3f}, "
          f"AUC={final_metrics['has_aftershocks_auc']:.3f}")
    print(f"   Tsunami Detection:     Acc={final_metrics['tsunami_accuracy']:.3f}, "
          f"P={final_metrics['tsunami_precision']:.3f}, "
          f"R={final_metrics['tsunami_recall']:.3f}, "
          f"F1={final_metrics['tsunami_f1']:.3f}, "
          f"AUC={final_metrics['tsunami_auc']:.3f}")
    print(f"   Foreshock Detection:   Acc={final_metrics['is_foreshock_accuracy']:.3f}, "
          f"P={final_metrics['is_foreshock_precision']:.3f}, "
          f"R={final_metrics['is_foreshock_recall']:.3f}, "
          f"F1={final_metrics['is_foreshock_f1']:.3f}, "
          f"AUC={final_metrics['is_foreshock_auc']:.3f}")
    
    print(f"\n   >>> AVERAGE F1: {avg_f1:.3f} <<<")
    
    print("\nLearned Physics Parameters:")
    print(f"   Gutenberg-Richter b-value: {final_metrics['physics_b_value']:.3f} (expected ~1.0)")
    print(f"   Omori p-value: {final_metrics['physics_p_value']:.3f} (expected ~1.0)")
    print(f"   Omori c-value: {final_metrics['physics_c_value']:.4f}")
    print(f"   Bath's ΔM: {final_metrics['physics_delta_m']:.3f} (expected ~1.2)")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'final_metrics': final_metrics,
        'processor_stats': processor.stats,
    }, 'piebm_earthquake_model.pt')
    
    print("\n✅ Model saved to piebm_earthquake_model.pt")
    
    return model, processor, df, history


if __name__ == "__main__":
    model, processor, df, history = main()