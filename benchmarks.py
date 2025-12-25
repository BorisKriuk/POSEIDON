#!/usr/bin/env python3
"""
Benchmarks: Compare PI-EBM against baseline and SOTA models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import json
import time
from typing import Dict, List, Tuple
from tqdm import tqdm

warnings.filterwarnings('ignore')

from model import ModelConfig, EarthquakeDataProcessor, EarthquakeDataset, PIEBM, Trainer


class LSTMBaseline(nn.Module):
    """LSTM baseline for sequence prediction"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class CNNBaseline(nn.Module):
    """CNN-only baseline without EBM components"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
    
    def forward(self, grid, event_features):
        spatial = self.conv(grid)
        combined = torch.cat([spatial, event_features], dim=1)
        return self.fc(combined)


class TransformerBaseline(nn.Module):
    """Transformer baseline"""
    
    def __init__(self, input_dim: int = 8, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def prepare_sklearn_data(grids: List[np.ndarray], event_features: List[np.ndarray], labels: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten data for sklearn models"""
    X_list = []
    y_list = []
    
    for i in range(len(grids)):
        grid_flat = grids[i].flatten()
        grid_stats = np.array([
            grids[i].mean(),
            grids[i].std(),
            grids[i].max(),
            grids[i][0].sum(),
            grids[i][1].max(),
            grids[i][2].sum()
        ])
        feat = event_features[i]
        X_list.append(np.concatenate([grid_stats, feat]))
        y_list.append([
            labels[i]['has_aftershocks'],
            labels[i]['tsunami'],
            labels[i]['is_foreshock']
        ])
    
    return np.array(X_list), np.array(y_list)


def evaluate_sklearn_model(model, X_train, y_train, X_val, y_val, task_idx: int, task_name: str) -> Dict:
    """Train and evaluate sklearn model"""
    start_time = time.time()
    model.fit(X_train, y_train[:, task_idx])
    train_time = time.time() - start_time
    
    start_time = time.time()
    y_pred = model.predict(X_val)
    inference_time = time.time() - start_time
    
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val[:, task_idx], y_prob)
    except:
        auc = 0.0
    
    return {
        'task': task_name,
        'accuracy': accuracy_score(y_val[:, task_idx], y_pred),
        'precision': precision_score(y_val[:, task_idx], y_pred, zero_division=0),
        'recall': recall_score(y_val[:, task_idx], y_pred, zero_division=0),
        'f1': f1_score(y_val[:, task_idx], y_pred, zero_division=0),
        'auc': auc,
        'train_time': train_time,
        'inference_time': inference_time
    }


def evaluate_pytorch_model(model, val_loader, device, task_names: List[str], is_piebm: bool = False) -> Dict:
    """Evaluate PyTorch model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch in val_loader:
            if is_piebm:
                grid = batch['grid'].to(device)
                event_features = batch['event_features'].to(device)
                outputs = model(grid, event_features)
                preds = torch.stack([
                    outputs['has_aftershocks'],
                    outputs['tsunami'],
                    outputs['is_foreshock']
                ], dim=1)
            elif isinstance(model, CNNBaseline):
                grid = batch['grid'].to(device)
                event_features = batch['event_features'].to(device)
                preds = model(grid, event_features)
            else:
                event_features = batch['event_features'].to(device)
                preds = model(event_features)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['labels'][:, 2:5].numpy())
    
    inference_time = time.time() - start_time
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    results = {}
    for i, task_name in enumerate(task_names):
        y_pred = (all_preds[:, i] > 0.5).astype(int)
        y_true = all_labels[:, i]
        
        try:
            auc = roc_auc_score(y_true, all_preds[:, i])
        except:
            auc = 0.0
        
        results[task_name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': auc
        }
    
    results['inference_time'] = inference_time
    return results


def train_pytorch_baseline(model, train_loader, val_loader, device, epochs: int = 30, is_cnn: bool = False):
    """Train PyTorch baseline model"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if is_cnn:
                grid = batch['grid'].to(device)
                event_features = batch['event_features'].to(device)
                outputs = model(grid, event_features)
            else:
                event_features = batch['event_features'].to(device)
                outputs = model(event_features)
            
            labels = batch['labels'][:, 2:5].to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                break
    
    train_time = time.time() - start_time
    return model, train_time


def run_benchmarks(csv_path: str = 'earthquake_dataset/earthquakes_clean.csv'):
    """Run all benchmarks"""
    
    print("=" * 80)
    print("EARTHQUAKE PREDICTION MODEL BENCHMARKS")
    print("=" * 80)
    
    config = ModelConfig()
    device = config.device
    print(f"\nDevice: {device}")
    
    print("\n[1/5] Loading data...")
    processor = EarthquakeDataProcessor(config)
    df = processor.load_data(csv_path)
    
    print("\n[2/5] Creating dataset...")
    grids, event_features, labels = processor.create_dataset(df, min_magnitude=4.5)
    
    if len(grids) < 100:
        grids, event_features, labels = processor.create_dataset(df, min_magnitude=4.0)
    
    dataset = EarthquakeDataset(grids, event_features, labels)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    X_train, y_train = prepare_sklearn_data(
        [grids[i] for i in train_indices],
        [event_features[i] for i in train_indices],
        [labels[i] for i in train_indices]
    )
    X_val, y_val = prepare_sklearn_data(
        [grids[i] for i in val_indices],
        [event_features[i] for i in val_indices],
        [labels[i] for i in val_indices]
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    task_names = ['has_aftershocks', 'tsunami', 'is_foreshock']
    all_results = []
    
    print("\n[3/5] Evaluating sklearn baselines...")
    
    sklearn_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced'),
        'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True)
    }
    
    for model_name, model in sklearn_models.items():
        print(f"   Training {model_name}...")
        for task_idx, task_name in enumerate(task_names):
            result = evaluate_sklearn_model(model, X_train_scaled, y_train, X_val_scaled, y_val, task_idx, task_name)
            result['model'] = model_name
            all_results.append(result)
    
    print("\n[4/5] Evaluating PyTorch baselines...")
    
    print("   Training LSTM...")
    lstm_model = LSTMBaseline(input_dim=8)
    lstm_model, lstm_train_time = train_pytorch_baseline(lstm_model, train_loader, val_loader, device)
    lstm_results = evaluate_pytorch_model(lstm_model, val_loader, device, task_names)
    for task_name in task_names:
        result = lstm_results[task_name].copy()
        result['model'] = 'LSTM'
        result['task'] = task_name
        result['train_time'] = lstm_train_time
        result['inference_time'] = lstm_results['inference_time'] / 3
        all_results.append(result)
    
    print("   Training Transformer...")
    transformer_model = TransformerBaseline(input_dim=8)
    transformer_model, transformer_train_time = train_pytorch_baseline(transformer_model, train_loader, val_loader, device)
    transformer_results = evaluate_pytorch_model(transformer_model, val_loader, device, task_names)
    for task_name in task_names:
        result = transformer_results[task_name].copy()
        result['model'] = 'Transformer'
        result['task'] = task_name
        result['train_time'] = transformer_train_time
        result['inference_time'] = transformer_results['inference_time'] / 3
        all_results.append(result)
    
    print("   Training CNN (no EBM)...")
    cnn_model = CNNBaseline(config)
    cnn_model, cnn_train_time = train_pytorch_baseline(cnn_model, train_loader, val_loader, device, is_cnn=True)
    cnn_results = evaluate_pytorch_model(cnn_model, val_loader, device, task_names)
    for task_name in task_names:
        result = cnn_results[task_name].copy()
        result['model'] = 'CNN (no EBM)'
        result['task'] = task_name
        result['train_time'] = cnn_train_time
        result['inference_time'] = cnn_results['inference_time'] / 3
        all_results.append(result)
    
    print("\n[5/5] Training and evaluating PI-EBM...")
    
    piebm_model = PIEBM(config)
    trainer = Trainer(piebm_model, config)
    
    piebm_start = time.time()
    trainer.train(train_loader, val_loader, epochs=config.epochs)
    piebm_train_time = time.time() - piebm_start
    
    piebm_results = evaluate_pytorch_model(piebm_model, val_loader, device, task_names, is_piebm=True)
    for task_name in task_names:
        result = piebm_results[task_name].copy()
        result['model'] = 'PI-EBM (Ours)'
        result['task'] = task_name
        result['train_time'] = piebm_train_time
        result['inference_time'] = piebm_results['inference_time'] / 3
        all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('benchmark_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    for task_name in task_names:
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print("-" * 70)
        task_df = results_df[results_df['task'] == task_name].sort_values('f1', ascending=False)
        print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
        print("-" * 70)
        for _, row in task_df.iterrows():
            print(f"{row['model']:<25} {row['accuracy']:>10.3f} {row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1']:>10.3f} {row['auc']:>10.3f}")
    
    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE ACROSS ALL TASKS")
    print("=" * 80)
    
    avg_results = results_df.groupby('model').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'auc': 'mean',
        'train_time': 'first',
        'inference_time': 'mean'
    }).sort_values('f1', ascending=False)
    
    print(f"\n{'Model':<25} {'Avg F1':>10} {'Avg AUC':>10} {'Train(s)':>12} {'Infer(s)':>12}")
    print("-" * 70)
    for model_name, row in avg_results.iterrows():
        print(f"{model_name:<25} {row['f1']:>10.3f} {row['auc']:>10.3f} {row['train_time']:>12.2f} {row['inference_time']:>12.4f}")
    
    piebm_f1 = avg_results.loc['PI-EBM (Ours)', 'f1']
    best_baseline_f1 = avg_results.drop('PI-EBM (Ours)')['f1'].max()
    improvement = ((piebm_f1 - best_baseline_f1) / best_baseline_f1) * 100
    
    print(f"\n✅ PI-EBM improvement over best baseline: {improvement:+.1f}%")
    
    summary = {
        'total_samples': len(dataset),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'models_evaluated': len(results_df['model'].unique()),
        'best_model': avg_results['f1'].idxmax(),
        'best_f1': avg_results['f1'].max(),
        'piebm_improvement': improvement,
        'device': device
    }
    
    with open('benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Results saved to benchmark_results.csv and benchmark_summary.json")
    
    return results_df, avg_results


if __name__ == "__main__":
    results_df, avg_results = run_benchmarks()