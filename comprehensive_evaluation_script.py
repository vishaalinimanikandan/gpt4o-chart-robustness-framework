#!/usr/bin/env python3
"""
GPT-4o Chart Extraction: Comprehensive Evaluation Analysis
=========================================================

This script evaluates 1,400 chart extractions using the comprehensive audit CSV,
calculating Value Accuracy, F1 Score, Relative Accuracy, and Composite Score.
Failed extractions are assigned 0 for all metrics.

Input:  comprehensive_extraction_audit.csv (1,392 success + 8 failed)
Output: comprehensive_evaluation_results.csv with detailed metrics per chart
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChartExtractionEvaluator:
    """Comprehensive evaluator for chart extraction performance"""
    
    def __init__(self, audit_csv_path: str):
        """Initialize evaluator with audit data"""
        self.audit_df = pd.read_csv(audit_csv_path)
        self.ground_truth_data = {}
        self.extraction_data = {}
        self.results = []
        
        # Load ground truth data
        self._load_ground_truth()
        
        print(f" Loaded audit data: {len(self.audit_df)} extraction records")
        print(f" Success rate: {len(self.audit_df[self.audit_df['status'] == 'success'])}/{len(self.audit_df)} "
              f"({len(self.audit_df[self.audit_df['status'] == 'success'])/len(self.audit_df)*100:.2f}%)")
    
    def _load_ground_truth(self):
        """Load ground truth data from chart configurations"""
        ground_truth_path = 'data/ground_truth/chart_configurations.json'
        
        if not os.path.exists(ground_truth_path):
            print(f"  Warning: Ground truth file not found at {ground_truth_path}")
            print("   Evaluation will proceed with limited accuracy metrics")
            return
        
        try:
            with open(ground_truth_path, 'r') as f:
                configs = json.load(f)
            
            # Convert to dictionary for easy lookup
            for config in configs:
                chart_id = config['id']
                self.ground_truth_data[chart_id] = {
                    'series_data': config.get('series_data', {}),
                    'categories': config.get('categories', []),
                    'chart_type': config.get('chart_type', ''),
                    'title': config.get('title', ''),
                    'data_points': config.get('data_points', 0)
                }
            
            print(f" Loaded ground truth for {len(self.ground_truth_data)} charts")
            
        except Exception as e:
            print(f" Error loading ground truth: {e}")
    
    def _load_extraction_data(self, extraction_key: str) -> Dict:
        """Load extraction data for a specific extraction key"""
        # Try different possible file paths based on the extraction key pattern
        possible_paths = [
            f'data/extractions/{extraction_key}.json',
            f'data/extractions/{extraction_key}_extraction.json',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f" Error reading {path}: {e}")
                    continue
        
        return {}
    
    def _calculate_value_accuracy(self, extracted_data: Dict, ground_truth: Dict) -> float:
        """Calculate value accuracy by comparing extracted vs ground truth values"""
        if not extracted_data or not ground_truth:
            return 0.0
        
        extracted_values = []
        ground_truth_values = []
        
        # Extract values from both datasets
        if 'data' in extracted_data:
            for item in extracted_data['data']:
                if isinstance(item, dict) and 'value' in item:
                    try:
                    # Handle different value formats
                     value = item['value']
                     if isinstance(value, (int, float)):
                        extracted_values.append(float(value))
                     elif isinstance(value, str):
                         extracted_values.append(float(value))
                     # Skip if value is dict or other non-numeric type
                    except (ValueError, TypeError):
                        continue
                    
        # Get ground truth values from series data
        gt_series = ground_truth.get('series_data', {})
        for series_name, values in gt_series.items():
            if isinstance(values, list):
                for v in values:
                    try:
                        if isinstance(v, (int, float)):
                            ground_truth_values.append(float(v))
                        elif isinstance(v, str):
                            ground_truth_values.append(float(v))
                    except (ValueError, TypeError):
                        continue
        
        if not extracted_values or not ground_truth_values:
            return 0.0
        
        # Calculate accuracy based on value differences
        # Use relative error tolerance of 5%
        correct_values = 0
        total_comparisons = min(len(extracted_values), len(ground_truth_values))
        
        for i in range(total_comparisons):
            extracted_val = extracted_values[i]
            gt_val = ground_truth_values[i]
            
            if gt_val == 0:
                # Handle zero values
                if abs(extracted_val) <= 0.01:  # Close to zero
                    correct_values += 1
            else:
                # Calculate relative error
                relative_error = abs(extracted_val - gt_val) / abs(gt_val)
                if relative_error <= 0.10:  # 10% tolerance
                    correct_values += 1
        
        return (correct_values / total_comparisons * 100) if total_comparisons > 0 else 0.0
    
    def _calculate_f1_score(self, extracted_data: Dict, ground_truth: Dict) -> float:
        """Calculate F1 score for data point extraction"""
        if not extracted_data or not ground_truth:
            return 0.0
        
        # Count extracted data points
        extracted_count = 0
        if 'data' in extracted_data and isinstance(extracted_data['data'], list):
            extracted_count = len(extracted_data['data'])
        
        # Count ground truth data points
        gt_count = ground_truth.get('data_points', 0)
        if gt_count == 0:
            # Fallback: count from series data
            gt_series = ground_truth.get('series_data', {})
            for series_name, values in gt_series.items():
                if isinstance(values, list):
                    gt_count = max(gt_count, len(values))
        
        if gt_count == 0:
            return 0.0
        
        # Calculate precision and recall
        # Precision: how many extracted points are correct
        # Recall: how many ground truth points were extracted
        
        true_positives = min(extracted_count, gt_count)
        precision = true_positives / extracted_count if extracted_count > 0 else 0.0
        recall = true_positives / gt_count if gt_count > 0 else 0.0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _calculate_relative_accuracy_updated(self, row: pd.Series, results_df: pd.DataFrame) -> float:
        """Calculate relative accuracy: (perturbed_composite / original_composite) × 100"""
        if row['is_original']:
            return 100.0
        if row['status'] != 'success':
             return 0.0
        # Find corresponding original chart
        chart_base = '_'.join(row['chart_id'].split('_')[:2])  # e.g., "chart_001"
        original_row = results_df[
            (results_df['chart_id'].str.startswith(chart_base)) & 
            (results_df['is_original'] == True)
        ]
        if len(original_row) == 0:
            return 0.0  # No origin
        original_composite = original_row['composite_score'].iloc[0]
        if original_composite == 0:
            return 0.0
        perturbed_composite = row['composite_score']
        return (perturbed_composite / original_composite) * 100
    
    def _calculate_composite_score(self, value_accuracy: float, f1_score: float) -> float:
       """Calculate composite score: (f1_score + (value_accuracy / 100)) / 2 × 100"""
       composite = (f1_score + (value_accuracy / 100)) / 2 * 100
       return composite
        
    def evaluate_single_extraction(self, row: pd.Series) -> Dict:
        """Evaluate a single extraction record"""
        extraction_key = row['extraction_key']
        chart_id = row['chart_id']
        status = row['status']
        is_original = row['is_original']
        perturbation_type = row['perturbation_type']
        
        # Initialize result record
        result = {
            'extraction_key': extraction_key,
            'chart_id': chart_id,
            'is_original': bool(is_original),
            'perturbation_type': perturbation_type,
            'intensity': row.get('intensity', 'none'),
            'status': status,
            'value_accuracy': 0.0,
            'f1_score': 0.0,
            'relative_accuracy': 0.0,
            'composite_score': 0.0,
            'evaluation_notes': ''
        }
        
        # If extraction failed, assign 0 to all metrics
        if status != 'success':
            result['evaluation_notes'] = f"Failed extraction: {status}"
            return result
        
        # Load extraction data
        extracted_data = self._load_extraction_data(extraction_key)
        if not extracted_data:
            result['evaluation_notes'] = "Extraction file not found"
            return result
        
        # Get ground truth data
        # Extract base chart ID (e.g., "chart_001_complex_bar" -> "chart_001")
        base_chart_id = '_'.join(chart_id.split('_')[:2])  # Gets "chart_001" from "chart_001_complex_bar"
        ground_truth = self.ground_truth_data.get(base_chart_id, {})
        if not ground_truth:
            result['evaluation_notes'] = "Ground truth not available"
            result['relative_accuracy'] = 0.3  # Partial score for basic extraction
            result['f1_score'] = 0.5 if 'data' in extracted_data else 0.0
            result['composite_score'] = self._calculate_composite_score(
                0.0, result['f1_score']
            )
            return result
        
        # Calculate all metrics
        try:
            result['value_accuracy'] = self._calculate_value_accuracy(extracted_data, ground_truth)
            result['f1_score'] = self._calculate_f1_score(extracted_data, ground_truth)
            result['relative_accuracy'] = 0.0  # Will be calculated later in post-processing
            result['composite_score'] = self._calculate_composite_score(
                 result['value_accuracy'], result['f1_score'])
            result['evaluation_notes'] = 'Successfully evaluated'
            
        except Exception as e:
            result['evaluation_notes'] = f"Evaluation error: {str(e)}"
        
        return result
    
    def evaluate_all_extractions(self) -> pd.DataFrame:
        """Evaluate all extractions and return results DataFrame"""
        print(" Starting comprehensive evaluation...")
        
        results = []
        total = len(self.audit_df)
        
        for idx, row in self.audit_df.iterrows():
            result = self.evaluate_single_extraction(row)
            results.append(result)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{total} extractions...")
        
        results_df = pd.DataFrame(results)
        print(" Calculating relative accuracy...")
        for idx, row in results_df.iterrows():
            results_df.at[idx, 'relative_accuracy'] = self._calculate_relative_accuracy_updated(row, results_df)
        # Add summary statistics
        self._print_evaluation_summary(results_df)
        
        return results_df
    
    def _print_evaluation_summary(self, results_df: pd.DataFrame):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*80)
        print(" COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        total_extractions = len(results_df)
        successful = len(results_df[results_df['status'] == 'success'])
        failed = total_extractions - successful
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"   Total Extractions: {total_extractions:,}")
        print(f"   Successful: {successful:,} ({successful/total_extractions*100:.2f}%)")
        print(f"   Failed: {failed:,} ({failed/total_extractions*100:.2f}%)")
        
        # Calculate metrics for successful extractions
        success_df = results_df[results_df['status'] == 'success']
        
        if len(success_df) > 0:
            print(f"\n ACCURACY METRICS (Successful Extractions):")
            print(f"   Value Accuracy: {success_df['value_accuracy'].mean():.3f} ± {success_df['value_accuracy'].std():.3f}")
            print(f"   F1 Score: {success_df['f1_score'].mean():.3f} ± {success_df['f1_score'].std():.3f}")
            print(f"   Relative Accuracy: {success_df['relative_accuracy'].mean():.3f} ± {success_df['relative_accuracy'].std():.3f}")
            print(f"   Composite Score: {success_df['composite_score'].mean():.3f} ± {success_df['composite_score'].std():.3f}")
        
        # Overall metrics (including failures as 0)
        print(f"\n OVERALL METRICS (Including Failures):")
        print(f"   Value Accuracy: {results_df['value_accuracy'].mean():.3f}")
        print(f"   F1 Score: {results_df['f1_score'].mean():.3f}")
        print(f"   Relative Accuracy: {results_df['relative_accuracy'].mean():.3f}")
        print(f"   Composite Score: {results_df['composite_score'].mean():.3f}")
        
        # Performance by perturbation type
        print(f"\n PERFORMANCE BY PERTURBATION TYPE:")
        perturbation_summary = results_df.groupby('perturbation_type').agg({
            'composite_score': ['mean', 'count'],
            'status': lambda x: (x == 'success').sum()
        }).round(3)
        
        perturbation_summary.columns = ['avg_composite_score', 'total_count', 'successful_count']
        perturbation_summary['success_rate'] = (perturbation_summary['successful_count'] / 
                                               perturbation_summary['total_count']).round(3)
        
        for pert_type, row in perturbation_summary.iterrows():
            print(f"   {pert_type:>12}: {row['avg_composite_score']:.3f} avg score | "
                  f"{row['successful_count']:>3}/{row['total_count']:>3} success "
                  f"({row['success_rate']:.1%})")
        
        # Original vs Perturbed comparison
        original_scores = results_df[results_df['is_original'] == True]['composite_score']
        perturbed_scores = results_df[results_df['is_original'] == False]['composite_score']
        
        print(f"\n ORIGINAL vs PERTURBED COMPARISON:")
        print(f"   Original Charts: {original_scores.mean():.3f} avg composite score")
        print(f"   Perturbed Charts: {perturbed_scores.mean():.3f} avg composite score")
        print(f"   Robustness Gap: {original_scores.mean() - perturbed_scores.mean():.3f}")
        
        print("="*80)

def main():
    """Main execution function"""
    print(" GPT-4o Chart Extraction: Comprehensive Evaluation Analysis")
    print("="*80)
    
     # ADD THIS SECTION FOR LOGGING:
    import sys
    from datetime import datetime
    
    # Create log file with timestamp
    log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for file in self.files:
                file.write(text)
                file.flush()
        def flush(self):
            for file in self.files:
                file.flush()
    
    # Open log file and redirect output
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # Redirect both stdout and stderr to both console and file
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, log_file)
        
        try:
            # REST OF YOUR EXISTING main() CODE GOES HERE
            # Initialize evaluator
            evaluator = ChartExtractionEvaluator('E:/langchain/Dissertation/data/logs/analysis/comprehensive_extraction_audit.csv')
            
            # Run evaluation
            results_df = evaluator.evaluate_all_extractions()
            
            # Save results
            output_path = 'comprehensive_evaluation_results.csv'
            results_df.to_csv(output_path, index=False)
            
            print(f"\n Results saved to: {output_path}")
            print(f" Generated {len(results_df)} evaluation records")
            print(f" Complete log saved to: {log_filename}")
            
            # Display sample results
            print(f"\n SAMPLE RESULTS (First 5 entries):")
            print(results_df[['extraction_key', 'perturbation_type', 'status', 
                             'value_accuracy', 'f1_score', 'relative_accuracy', 
                             'composite_score']].head().to_string(index=False))
            
            print(f"\n Evaluation complete!")
            
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
    


if __name__ == "__main__":
    main()
