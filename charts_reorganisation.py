#!/usr/bin/env python3
"""
Chart Files Reorganization Script
=================================
Reorganizes raw charts and perturbations into a structured hierarchy
for better analysis.

Structure:
- Raw charts: complexity → chart_type
- Perturbations: perturbation_type → complexity → chart_type
"""

import os
import shutil
from pathlib import Path
import re
from collections import defaultdict

def create_directory_structure():
    """Create the new organized directory structure"""
    base_dirs = [
        "data/raw_charts_organized",
        "data/perturbations_organized"
    ]
    
    complexities = ['complex', 'medium', 'advanced']
    chart_types = ['bar', 'pie', 'line', 'scatter', 'area']
    
    # Create raw charts structure
    for complexity in complexities:
        for chart_type in chart_types:
            path = f"data/raw_charts_organized/{complexity}/{chart_type}"
            os.makedirs(path, exist_ok=True)
    
    print("✓ Created raw_charts_organized structure")
    return complexities, chart_types

def organize_raw_charts():
    """Organize raw charts by complexity and type"""
    print("\n=== ORGANIZING RAW CHARTS ===")
    
    raw_charts_dir = Path("data/raw_charts")
    if not raw_charts_dir.exists():
        print(" data/raw_charts directory not found!")
        return
    
    chart_files = list(raw_charts_dir.glob("*.png"))
    print(f"Found {len(chart_files)} raw charts")
    
    organized_count = 0
    errors = []
    
    for chart_file in chart_files:
        # Parse filename: chart_001_complex_bar.png
        filename = chart_file.stem  # without .png
        parts = filename.split('_')
        
        if len(parts) >= 4:
            chart_id = f"{parts[0]}_{parts[1]}"  # chart_001
            complexity = parts[2]  # complex/medium/advanced
            chart_type = parts[3]  # bar/pie/line/scatter/area
            
            # Validate
            valid_complexities = ['complex', 'medium', 'advanced']
            valid_types = ['bar', 'pie', 'line', 'scatter', 'area']
            
            if complexity in valid_complexities and chart_type in valid_types:
                # Copy to new location
                dest_dir = f"data/raw_charts_organized/{complexity}/{chart_type}"
                dest_path = Path(dest_dir) / chart_file.name
                
                try:
                    shutil.copy2(chart_file, dest_path)
                    organized_count += 1
                except Exception as e:
                    errors.append(f"Error copying {chart_file.name}: {str(e)}")
            else:
                errors.append(f"Invalid format: {chart_file.name} - complexity: {complexity}, type: {chart_type}")
        else:
            errors.append(f"Cannot parse: {chart_file.name}")
    
    print(f"✓ Organized {organized_count} raw charts")
    if errors:
        print(f"  {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    
    return organized_count

def organize_perturbations():
    """Organize perturbations by type, complexity, and chart type"""
    print("\n=== ORGANIZING PERTURBATIONS ===")
    
    perturbations_dir = Path("data/perturbations")
    if not perturbations_dir.exists():
        print(" data/perturbations directory not found!")
        return
    
    pert_files = list(perturbations_dir.glob("*.png"))
    print(f"Found {len(pert_files)} perturbation files")
    
    # First, identify all perturbation types
    perturbation_types = set()
    for pert_file in pert_files:
        # Parse: chart_001_complex_bar_axis_degradation_medium.png
        parts = pert_file.stem.split('_')
        if len(parts) >= 6:
            # Perturbation type might be multi-word (axis_degradation, gaussian_blur, etc.)
            # Strategy: everything between chart_type and intensity level
            chart_type_idx = 3
            intensity_idx = -1  # last part is usually intensity
            
            pert_type_parts = parts[chart_type_idx + 1:intensity_idx]
            pert_type = '_'.join(pert_type_parts)
            perturbation_types.add(pert_type)
    
    print(f"Found {len(perturbation_types)} perturbation types:")
    for pt in sorted(perturbation_types):
        print(f"  - {pt}")
    
    # Create directory structure for each perturbation type
    complexities = ['complex', 'medium', 'advanced']
    chart_types = ['bar', 'pie', 'line', 'scatter', 'area']
    
    for pert_type in perturbation_types:
        for complexity in complexities:
            for chart_type in chart_types:
                path = f"data/perturbations_organized/{pert_type}/{complexity}/{chart_type}"
                os.makedirs(path, exist_ok=True)
    
    # Organize files
    organized_count = 0
    errors = []
    
    for pert_file in pert_files:
        filename = pert_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 6:
            chart_id = f"{parts[0]}_{parts[1]}"  # chart_001
            complexity = parts[2]  # complex/medium/advanced
            chart_type = parts[3]  # bar/pie/line/scatter/area
            intensity = parts[-1]  # low/medium/high
            
            # Extract perturbation type (everything between chart_type and intensity)
            pert_type_parts = parts[4:-1]
            pert_type = '_'.join(pert_type_parts)
            
            # Validate
            if (complexity in complexities and 
                chart_type in chart_types and 
                pert_type in perturbation_types):
                
                # Copy to new location
                dest_dir = f"data/perturbations_organized/{pert_type}/{complexity}/{chart_type}"
                dest_path = Path(dest_dir) / pert_file.name
                
                try:
                    shutil.copy2(pert_file, dest_path)
                    organized_count += 1
                except Exception as e:
                    errors.append(f"Error copying {pert_file.name}: {str(e)}")
            else:
                errors.append(f"Invalid format: {pert_file.name}")
        else:
            errors.append(f"Cannot parse: {pert_file.name} - only {len(parts)} parts")
    
    print(f"✓ Organized {organized_count} perturbation files")
    if errors:
        print(f"  {len(errors)} errors:")
        for error in errors[:5]:
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more")
    
    return organized_count, perturbation_types

def generate_summary_report():
    """Generate a summary of the organization"""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report_lines = []
    report_lines.append("FILE ORGANIZATION SUMMARY")
    report_lines.append("=" * 50)
    
    # Count raw charts
    raw_organized_dir = Path("data/raw_charts_organized")
    if raw_organized_dir.exists():
        report_lines.append("\nRAW CHARTS:")
        for complexity in ['complex', 'medium', 'advanced']:
            complexity_total = 0
            report_lines.append(f"\n{complexity.upper()}:")
            for chart_type in ['bar', 'pie', 'line', 'scatter', 'area']:
                chart_dir = raw_organized_dir / complexity / chart_type
                if chart_dir.exists():
                    count = len(list(chart_dir.glob("*.png")))
                    complexity_total += count
                    report_lines.append(f"  {chart_type}: {count} charts")
            report_lines.append(f"  Total: {complexity_total} charts")
    
    # Count perturbations
    pert_organized_dir = Path("data/perturbations_organized")
    if pert_organized_dir.exists():
        report_lines.append("\n\nPERTURBATIONS:")
        
        for pert_type in sorted([d.name for d in pert_organized_dir.iterdir() if d.is_dir()]):
            pert_total = 0
            report_lines.append(f"\n{pert_type.upper()}:")
            
            for complexity in ['complex', 'medium', 'advanced']:
                complexity_dir = pert_organized_dir / pert_type / complexity
                if complexity_dir.exists():
                    complexity_count = sum(len(list((complexity_dir / ct).glob("*.png"))) 
                                         for ct in ['bar', 'pie', 'line', 'scatter', 'area'] 
                                         if (complexity_dir / ct).exists())
                    pert_total += complexity_count
                    report_lines.append(f"  {complexity}: {complexity_count} files")
            
            report_lines.append(f"  Total: {pert_total} files")
    
    # Save report
    report_text = '\n'.join(report_lines)
    with open("data/organization_summary.txt", 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print("\n✓ Summary saved to: data/organization_summary.txt")

def main():
    """Main execution"""
    print("="*60)
    print("CHART FILES REORGANIZATION")
    print("="*60)
    
    # Create directory structure
    create_directory_structure()
    
    # Organize raw charts
    raw_count = organize_raw_charts()
    
    # Organize perturbations
    pert_count, pert_types = organize_perturbations()
    
    # Generate summary
    generate_summary_report()
    
    print("\n REORGANIZATION COMPLETE!")
    print(f"   Raw charts organized: {raw_count}")
    print(f"   Perturbations organized: {pert_count}")
    print("\nNew structure created in:")
    print("   - data/raw_charts_organized/")
    print("   - data/perturbations_organized/")

if __name__ == "__main__":
    main()