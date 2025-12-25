#!/usr/bin/env python3
"""
try.py - Earthquake Dataset Visualization & Statistics
Visualizes 3M+ earthquake records on world map with comprehensive statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️  Cartopy not installed. Using basic matplotlib maps.")
    print("   Install with: pip install cartopy")

def load_data():
    """Load the earthquake dataset"""
    print("📂 Loading earthquake data...")
    
    # Search patterns - check folder and current directory
    search_patterns = [
        'earthquake_dataset/*.csv',
        'earthquake_dataset/**/*.csv',
        'data/*.csv',
        '*.csv',
    ]
    
    csv_files = []
    for pattern in search_patterns:
        csv_files.extend(glob.glob(pattern, recursive=True))
    
    if not csv_files:
        print("❌ No CSV files found!")
        print("   Searched in:", search_patterns)
        return None
    
    print(f"   📁 Found {len(csv_files)} CSV file(s)")
    
    # If multiple files, concatenate them
    if len(csv_files) == 1:
        print(f"   📄 Loading: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])
    else:
        # Show what we found
        for f in csv_files[:10]:
            size_mb = os.path.getsize(f) / (1024*1024)
            print(f"      • {f} ({size_mb:.1f} MB)")
        if len(csv_files) > 10:
            print(f"      ... and {len(csv_files) - 10} more files")
        
        print(f"\n   ⏳ Concatenating {len(csv_files)} files...")
        dfs = []
        for f in csv_files:
            try:
                temp_df = pd.read_csv(f)
                dfs.append(temp_df)
            except Exception as e:
                print(f"      ⚠️ Skipping {f}: {e}")
        
        df = pd.concat(dfs, ignore_index=True)
    
    print(f"   ✅ Loaded {len(df):,} total records")
    
    # Show columns
    print(f"   📋 Columns: {list(df.columns)}")
    
    # Parse time if needed
    if 'time' in df.columns:
        print("   ⏳ Parsing timestamps...")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if 'year' not in df.columns:
            df['year'] = df['time'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['time'].dt.month
    
    # Check for required columns
    required = ['latitude', 'longitude', 'magnitude']
    missing = [col for col in required if col not in df.columns]
    if missing:
        # Try alternate column names
        alt_names = {
            'latitude': ['lat', 'Latitude', 'LAT'],
            'longitude': ['lon', 'long', 'Longitude', 'LON'],
            'magnitude': ['mag', 'Magnitude', 'MAG']
        }
        for col in missing:
            for alt in alt_names.get(col, []):
                if alt in df.columns:
                    df[col] = df[alt]
                    print(f"   🔄 Mapped {alt} -> {col}")
                    break
    
    # Final check
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return None
    
    # Calculate energy if not present
    if 'energy_joules' not in df.columns:
        print("   ⚡ Calculating energy from magnitude...")
        df['energy_joules'] = 10 ** (1.5 * df['magnitude'] + 4.8)
    
    return df

def print_statistics(df):
    """Print comprehensive dataset statistics"""
    print("\n" + "="*70)
    print("📊 DATASET STATISTICS")
    print("="*70)
    
    print(f"\n📈 RECORD COUNTS")
    print(f"   Total earthquakes:     {len(df):>12,}")
    print(f"   Date range:            {int(df['year'].min())} - {int(df['year'].max())}")
    print(f"   Years covered:         {int(df['year'].max() - df['year'].min() + 1)}")
    
    print(f"\n🎚️  MAGNITUDE DISTRIBUTION")
    print(f"   Minimum:               {df['magnitude'].min():>12.2f}")
    print(f"   Maximum:               {df['magnitude'].max():>12.2f}")
    print(f"   Mean:                  {df['magnitude'].mean():>12.2f}")
    print(f"   Median:                {df['magnitude'].median():>12.2f}")
    print(f"   Std Dev:               {df['magnitude'].std():>12.2f}")
    
    print(f"\n🔢 MAGNITUDE BRACKETS")
    brackets = [
        (0, 2, "Micro (M<2)"),
        (2, 3, "Minor (M2-3)"),
        (3, 4, "Light (M3-4)"),
        (4, 5, "Moderate (M4-5)"),
        (5, 6, "Strong (M5-6)"),
        (6, 7, "Major (M6-7)"),
        (7, 8, "Great (M7-8)"),
        (8, 10, "Massive (M8+)")
    ]
    
    for low, high, label in brackets:
        count = len(df[(df['magnitude'] >= low) & (df['magnitude'] < high)])
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:20s} {count:>10,} ({pct:>5.1f}%) {bar}")
    
    print(f"\n📍 DEPTH DISTRIBUTION (km)")
    if 'depth' in df.columns:
        print(f"   Shallow (0-70km):      {len(df[df['depth'] <= 70]):>12,}")
        print(f"   Intermediate (70-300): {len(df[(df['depth'] > 70) & (df['depth'] <= 300)]):>12,}")
        print(f"   Deep (300+ km):        {len(df[df['depth'] > 300]):>12,}")
        print(f"   Maximum depth:         {df['depth'].max():>12.1f} km")
    else:
        print("   ⚠️ No depth column found")
    
    print(f"\n🌐 GEOGRAPHIC COVERAGE")
    print(f"   Latitude range:        {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
    print(f"   Longitude range:       {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
    
    if 'tsunami' in df.columns:
        print(f"\n🌊 TSUNAMI EVENTS")
        tsunami_count = df['tsunami'].sum()
        print(f"   Tsunami warnings:      {int(tsunami_count):>12,}")
        print(f"   Percentage:            {tsunami_count/len(df)*100:>12.3f}%")
    
    print(f"\n⚡ ENERGY STATISTICS")
    print(f"   Total energy released: {df['energy_joules'].sum():>12.2e} Joules")
    print(f"   Max single event:      {df['energy_joules'].max():>12.2e} Joules")
    hiroshima = df['energy_joules'].sum() / 6.3e13
    print(f"   Total (Hiroshima eq.): {hiroshima:>12,.0f} bombs")
    
    if 'net' in df.columns:
        print(f"\n📡 REPORTING NETWORKS")
        top_networks = df['net'].value_counts().head(10)
        print("   Top 10 networks:")
        for net, count in top_networks.items():
            pct = count / len(df) * 100
            print(f"      {str(net):6s} {count:>10,} ({pct:>5.1f}%)")
    
    print(f"\n📅 YEARLY DISTRIBUTION")
    yearly = df.groupby('year').size()
    print(f"   Most active year:      {int(yearly.idxmax())} ({yearly.max():,} quakes)")
    print(f"   Least active year:     {int(yearly.idxmin())} ({yearly.min():,} quakes)")
    print(f"   Average per year:      {yearly.mean():,.0f}")
    
    print(f"\n🏆 TOP 10 LARGEST EARTHQUAKES")
    cols_to_show = ['time', 'magnitude', 'depth', 'place'] if 'place' in df.columns else ['time', 'magnitude', 'depth']
    cols_available = [c for c in cols_to_show if c in df.columns]
    top10 = df.nlargest(10, 'magnitude')[cols_available].reset_index(drop=True)
    
    for i, row in top10.iterrows():
        place = str(row.get('place', 'Unknown'))[:40] if 'place' in row else 'Unknown'
        time_str = str(row.get('time', 'Unknown'))[:10]
        depth = row.get('depth', 0)
        print(f"   {i+1:2d}. M{row['magnitude']:.1f} | {time_str} | {depth:.0f}km | {place}")
    
    print("\n" + "="*70)

def create_visualizations(df):
    """Create all visualization plots"""
    
    print("\n🎨 Generating visualizations...")
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. World Map
    print("   📍 Creating world map...")
    if HAS_CARTOPY:
        ax1 = fig.add_subplot(4, 2, 1, projection=ccrs.Robinson())
        ax1.set_global()
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none')
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        sample = df.sample(min(50000, len(df)))
        scatter = ax1.scatter(
            sample['longitude'], sample['latitude'],
            c=sample['magnitude'], cmap='YlOrRd',
            s=0.5, alpha=0.3, transform=ccrs.PlateCarree(),
            vmin=0, vmax=8
        )
        plt.colorbar(scatter, ax=ax1, label='Magnitude', shrink=0.6)
        ax1.set_title(f'Global Earthquake Distribution (n={len(df):,})', fontsize=12, fontweight='bold')
    else:
        ax1 = fig.add_subplot(4, 2, 1)
        sample = df.sample(min(50000, len(df)))
        scatter = ax1.scatter(
            sample['longitude'], sample['latitude'],
            c=sample['magnitude'], cmap='YlOrRd',
            s=0.5, alpha=0.3, vmin=0, vmax=8
        )
        plt.colorbar(scatter, ax=ax1, label='Magnitude')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Global Earthquake Distribution (n={len(df):,})', fontsize=12, fontweight='bold')
        ax1.set_xlim(-180, 180)
        ax1.set_ylim(-90, 90)
    
    # 2. Magnitude Histogram
    print("   📊 Creating magnitude histogram...")
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.hist(df['magnitude'], bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax2.set_xlabel('Magnitude', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Magnitude Distribution', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.axvline(df['magnitude'].mean(), color='red', linestyle='--', label=f'Mean: {df["magnitude"].mean():.2f}')
    ax2.axvline(df['magnitude'].median(), color='orange', linestyle='--', label=f'Median: {df["magnitude"].median():.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Depth vs Magnitude
    print("   🔬 Creating depth vs magnitude plot...")
    ax3 = fig.add_subplot(4, 2, 3)
    if 'depth' in df.columns:
        sample = df.sample(min(20000, len(df)))
        ax3.scatter(sample['magnitude'], sample['depth'], alpha=0.1, s=1, c='steelblue')
        ax3.set_xlabel('Magnitude', fontsize=11)
        ax3.set_ylabel('Depth (km)', fontsize=11)
        ax3.set_title('Depth vs Magnitude', fontsize=12, fontweight='bold')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No depth data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Yearly Count
    print("   📅 Creating yearly trends...")
    ax4 = fig.add_subplot(4, 2, 4)
    yearly = df.groupby('year').size()
    ax4.bar(yearly.index, yearly.values, color='steelblue', alpha=0.8)
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Number of Earthquakes', fontsize=11)
    ax4.set_title('Earthquakes Per Year', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Density Heatmap
    print("   🗺️  Creating density heatmap...")
    ax5 = fig.add_subplot(4, 2, 5)
    heatmap, xedges, yedges = np.histogram2d(
        df['longitude'], df['latitude'], 
        bins=[360, 180], 
        range=[[-180, 180], [-90, 90]]
    )
    heatmap_masked = np.ma.masked_where(heatmap.T == 0, heatmap.T)
    im = ax5.imshow(
        heatmap_masked, origin='lower', 
        extent=[-180, 180, -90, 90],
        cmap='hot', norm=LogNorm(vmin=1, vmax=heatmap.max()),
        aspect='auto'
    )
    plt.colorbar(im, ax=ax5, label='Earthquake Count (log scale)')
    ax5.set_xlabel('Longitude', fontsize=11)
    ax5.set_ylabel('Latitude', fontsize=11)
    ax5.set_title('Earthquake Density Heatmap', fontsize=12, fontweight='bold')
    
    # 6. Monthly Pattern
    print("   📆 Creating monthly patterns...")
    ax6 = fig.add_subplot(4, 2, 6)
    if 'month' in df.columns:
        monthly = df.groupby('month').size()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax6.bar(range(1, 13), [monthly.get(i, 0) for i in range(1, 13)], color='teal', alpha=0.8)
        ax6.set_xticks(range(1, 13))
        ax6.set_xticklabels(months, rotation=45)
        ax6.set_xlabel('Month', fontsize=11)
        ax6.set_ylabel('Total Earthquakes (1990-2019)', fontsize=11)
        ax6.set_title('Seasonal Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Cumulative Energy
    print("   ⚡ Creating energy release chart...")
    ax7 = fig.add_subplot(4, 2, 7)
    yearly_energy = df.groupby('year')['energy_joules'].sum().cumsum()
    ax7.fill_between(yearly_energy.index, yearly_energy.values / 1e18, color='crimson', alpha=0.6)
    ax7.plot(yearly_energy.index, yearly_energy.values / 1e18, color='darkred', linewidth=2)
    ax7.set_xlabel('Year', fontsize=11)
    ax7.set_ylabel('Cumulative Energy (×10¹⁸ Joules)', fontsize=11)
    ax7.set_title('Cumulative Seismic Energy Release', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Large Events by Year
    print("   🌋 Creating large event trends...")
    ax8 = fig.add_subplot(4, 2, 8)
    large_yearly = df[df['magnitude'] >= 6].groupby('year').size()
    ax8.bar(large_yearly.index, large_yearly.values, color='crimson', alpha=0.8, label='M6+')
    ax8.set_xlabel('Year', fontsize=11)
    ax8.set_ylabel('Number of M6+ Earthquakes', fontsize=11)
    ax8.set_title('Major Earthquakes (M6+) Per Year', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_file = 'earthquake_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Saved visualization to: {output_file}")
    plt.show()

def create_large_quake_map(df):
    """Create map of M6+ earthquakes"""
    
    print("\n🌋 Creating large earthquake map (M6+)...")
    
    large_quakes = df[df['magnitude'] >= 6].copy()
    print(f"   Found {len(large_quakes):,} earthquakes with M ≥ 6.0")
    
    fig = plt.figure(figsize=(16, 10))
    
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        sizes = (large_quakes['magnitude'] - 5.5) ** 3 * 5
        scatter = ax.scatter(
            large_quakes['longitude'], large_quakes['latitude'],
            c=large_quakes['magnitude'], cmap='YlOrRd',
            s=sizes, alpha=0.6, transform=ccrs.PlateCarree(),
            vmin=6, vmax=9, edgecolors='black', linewidths=0.3
        )
        plt.colorbar(scatter, ax=ax, label='Magnitude', shrink=0.6)
        
        mega_quakes = large_quakes[large_quakes['magnitude'] >= 8]
        for _, quake in mega_quakes.iterrows():
            ax.plot(quake['longitude'], quake['latitude'], 'r*', 
                   markersize=20, transform=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(1, 1, 1)
        sizes = (large_quakes['magnitude'] - 5.5) ** 3 * 5
        scatter = ax.scatter(
            large_quakes['longitude'], large_quakes['latitude'],
            c=large_quakes['magnitude'], cmap='YlOrRd',
            s=sizes, alpha=0.6, vmin=6, vmax=9,
            edgecolors='black', linewidths=0.3
        )
        plt.colorbar(scatter, ax=ax, label='Magnitude')
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        mega_quakes = large_quakes[large_quakes['magnitude'] >= 8]
        ax.scatter(mega_quakes['longitude'], mega_quakes['latitude'], 
                  c='red', marker='*', s=300, zorder=10)
    
    ax.set_title(f'Major Earthquakes M6+ (1990-2019) | n={len(large_quakes):,} | ★ = M8+', 
                fontsize=14, fontweight='bold')
    
    output_file = 'large_earthquakes_map.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   ✅ Saved to: {output_file}")
    plt.show()

def main():
    """Main execution"""
    print("="*70)
    print("🌍 EARTHQUAKE DATASET VISUALIZATION & ANALYSIS")
    print("="*70)
    
    df = load_data()
    if df is None:
        return
    
    print_statistics(df)
    create_visualizations(df)
    create_large_quake_map(df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print("\n📁 Output files created:")
    print("   • earthquake_analysis.png    - Main dashboard")
    print("   • large_earthquakes_map.png  - M6+ events map")
    print("\n🎯 Ready for ML pipeline!")

if __name__ == "__main__":
    main()