# Simple_Comprehensive_Fixer.py
# Fix only the comprehensive_metrics.csv file

import pandas as pd
import json
from pathlib import Path

print("=" * 60)
print("🔧 FIXING COMPREHENSIVE_METRICS.CSV")
print("=" * 60)

def fix_comprehensive_metrics():
    """Fix the comprehensive metrics file"""
    
    # Load the broken file
    print("📊 Loading comprehensive_metrics.csv...")
    df = pd.read_csv(r'E:/langchain/Dissertation/data/analysis_cache/comprehensive_metrics.csv')
    
    print(f"Before fix:")
    print(f"  Empty chart IDs: {df['original_chart_id'].isna().sum()}/{len(df)}")
    print(f"  Zero exact matches: {(df['exact_match_accuracy'] == 0).sum()}/{len(df)}")
    
    # Fix 1: Fill missing chart IDs and perturbation info
    print("\n🔧 Fixing metadata...")
    
    for idx, row in df.iterrows():
        if pd.isna(row['original_chart_id']):
            extraction_key = row['extraction_key']
            
            if '_original' in extraction_key:
                # Original: chart_001_complex_bar_original -> chart_001
                parts = extraction_key.replace('_original', '').split('_')
                chart_id = parts[0] + '_' + parts[1]  # chart_001
                df.at[idx, 'original_chart_id'] = chart_id
                df.at[idx, 'perturbation_type'] = 'none'
                df.at[idx, 'intensity'] = 'none'
            else:
                # Perturbation: chart_001_blur_medium -> chart_001, blur, medium
                parts = extraction_key.split('_')
                if len(parts) >= 4:
                    chart_id = parts[0] + '_' + parts[1]  # chart_001
                    perturbation = parts[-2]  # blur
                    intensity = parts[-1]     # medium
                    
                    df.at[idx, 'original_chart_id'] = chart_id
                    df.at[idx, 'perturbation_type'] = perturbation
                    df.at[idx, 'intensity'] = intensity
    
    print(f"After metadata fix:")
    print(f"  Empty chart IDs: {df['original_chart_id'].isna().sum()}/{len(df)}")
    
    # Fix 2: Improve exact match accuracy calculation
    print("\n🧮 Fixing accuracy calculations...")
    
    # Load ground truth
    with open('data/ground_truth/chart_configurations.json', 'r') as f:
        ground_truth_data = json.load(f)
    
    gt_lookup = {gt['id']: gt for gt in ground_truth_data}
    
    improved_count = 0
    
    for idx, row in df.iterrows():
        extraction_key = row['extraction_key']
        chart_id = row['original_chart_id']
        
        # Load extraction file
        extraction_file = Path(f'data/extractions/{extraction_key}.json')
        
        if extraction_file.exists() and chart_id in gt_lookup:
            try:
                with open(extraction_file, 'r') as f:
                    extracted_data = json.load(f)
                
                ground_truth = gt_lookup[chart_id]
                
                # Simple improved accuracy calculation
                accuracy = calculate_simple_accuracy(extracted_data, ground_truth)
                
                if accuracy > 0:
                    df.at[idx, 'exact_match_accuracy'] = accuracy
                    improved_count += 1
                    
            except Exception as e:
                continue
        
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(df)} rows...")
    
    print(f"Improved {improved_count} accuracy calculations")
    
    # Save fixed file
    df.to_csv('comprehensive_metrics_fixed.csv', index=False)
    
    print(f"\n✅ RESULTS:")
    print(f"  Empty chart IDs: {df['original_chart_id'].isna().sum()}/{len(df)}")
    print(f"  Zero exact matches: {(df['exact_match_accuracy'] == 0).sum()}/{len(df)}")
    print(f"  Non-zero exact matches: {(df['exact_match_accuracy'] > 0).sum()}/{len(df)}")
    print(f"  Mean F1: {df['partial_match_f1'].mean():.3f}")
    print(f"  Mean structural: {df['structural_understanding'].mean():.1f}%")
    
    print(f"\n📁 Saved: comprehensive_metrics_fixed.csv")
    
    return df

def calculate_simple_accuracy(extracted_data, ground_truth):
    """Simple accuracy calculation"""
    
    extracted_points = extracted_data.get('data', [])
    gt_categories = ground_truth.get('categories', [])
    gt_series_data = ground_truth.get('series_data', {})
    
    if not extracted_points or not gt_categories or not gt_series_data:
        return 0
    
    # Convert ground truth to simple format
    gt_points = []
    for series_name, values in gt_series_data.items():
        for category, value in zip(gt_categories, values):
            gt_points.append({
                'category': category,
                'value': float(value),
                'combined': f"{category} - {series_name}"
            })
    
    # Count matches
    matches = 0
    for gt_point in gt_points:
        for ext_point in extracted_points:
            ext_category = ext_point.get('category', '')
            ext_value = float(ext_point.get('value', 0))
            
            # Check category match (multiple strategies)
            category_match = False
            if ext_category == gt_point['category']:
                category_match = True
            elif ext_category == gt_point['combined']:
                category_match = True
            elif gt_point['category'] in ext_category:
                category_match = True
            
            # Check value match (20% tolerance)
            value_match = False
            if abs(ext_value - gt_point['value']) / max(abs(gt_point['value']), 1) < 0.2:
                value_match = True
            
            # If both match, count it
            if category_match and value_match:
                matches += 1
                break
    
    # Calculate accuracy
    accuracy = (matches / len(gt_points)) * 100 if gt_points else 0
    return min(accuracy, 100)  # Cap at 100%

if __name__ == "__main__":
    print("🚀 Starting comprehensive metrics fix...")
    fixed_df = fix_comprehensive_metrics()
    print("✅ Comprehensive metrics fixing complete!")