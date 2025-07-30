#!/usr/bin/env python
# coding: utf-8

# # Diagnostic Script - Why are all metrics showing 0%?

import json
import pandas as pd
from pathlib import Path
import random

# Set paths
EXTRACTION_PATH = Path(r"E:\langchain\Dissertation\data\extractions")
GROUND_TRUTH_PATH = Path(r"E:\langchain\Dissertation\data\ground_truth")

print("EXTRACTION DIAGNOSTIC ANALYSIS")
print("=" * 60)

# ## 1. Check extraction file structure

print("\n1. CHECKING EXTRACTION FILES:")
print("-" * 40)

# Get sample extraction files
extraction_files = list(EXTRACTION_PATH.glob("*.json"))
print(f"Total extraction files found: {len(extraction_files)}")

# Show sample filenames
print("\nSample extraction filenames:")
for f in extraction_files[:5]:
    print(f"  - {f.name}")

# ## 2. Examine extraction data structure

print("\n2. EXAMINING EXTRACTION DATA STRUCTURE:")
print("-" * 40)

# Load a few random extractions
sample_extractions = random.sample(extraction_files, min(3, len(extraction_files)))

for i, file_path in enumerate(sample_extractions):
    print(f"\nSample {i+1}: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Keys in extraction: {list(data.keys())}")
    
    if 'data' in data:
        print(f"  Data type: {type(data['data'])}")
        if data['data']:
            if isinstance(data['data'], list):
                print(f"  Data length: {len(data['data'])}")
                if len(data['data']) > 0:
                    print(f"  First data item: {data['data'][0]}")
            elif isinstance(data['data'], dict):
                print(f"  Data keys: {list(data['data'].keys())[:5]}...")
                first_key = list(data['data'].keys())[0] if data['data'] else None
                if first_key:
                    print(f"  Sample data: {first_key}: {data['data'][first_key]}")
        else:
            print("  ⚠️  Data field is empty!")
    else:
        print("  ⚠️  No 'data' field found!")
    
    if 'chart_type' in data:
        print(f"  Chart type: {data['chart_type']}")
    
    print(f"  All fields: {json.dumps(data, indent=2)[:500]}...")

# ## 3. Check ground truth structure

print("\n\n3. CHECKING GROUND TRUTH STRUCTURE:")
print("-" * 40)

# Check for chart_configurations.json
config_file = GROUND_TRUTH_PATH / "chart_configurations.json"
if config_file.exists():
    print("✓ Found chart_configurations.json")
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    print(f"  Total configurations: {len(configs)}")
    
    # Show sample configuration
    if configs:
        sample_config = configs[0]
        print(f"\nSample ground truth structure:")
        print(f"  ID: {sample_config.get('id', 'N/A')}")
        print(f"  Keys: {list(sample_config.keys())}")
        
        if 'categories' in sample_config:
            print(f"  Categories: {sample_config['categories'][:3]}...")
        if 'values' in sample_config:
            print(f"  Values type: {type(sample_config['values'])}")
            if isinstance(sample_config['values'], list):
                print(f"  Values sample: {sample_config['values'][:3]}...")
else:
    print("✗ chart_configurations.json not found!")
    
    # Check for individual ground truth files
    gt_files = list(GROUND_TRUTH_PATH.glob("*.json"))
    print(f"\nIndividual ground truth files: {len(gt_files)}")
    
    if gt_files:
        sample_gt = gt_files[0]
        print(f"\nSample file: {sample_gt.name}")
        with open(sample_gt, 'r') as f:
            gt_data = json.load(f)
        print(f"  Structure: {json.dumps(gt_data, indent=2)[:300]}...")

# ## 4. Test data matching

print("\n\n4. TESTING DATA MATCHING:")
print("-" * 40)

# Load one extraction and its ground truth
test_extraction_file = extraction_files[0]
print(f"\nTest file: {test_extraction_file.name}")

# Extract chart ID
chart_id = test_extraction_file.stem.split('_')[0] + '_' + test_extraction_file.stem.split('_')[1]
print(f"Chart ID: {chart_id}")

# Load extraction
with open(test_extraction_file, 'r') as f:
    test_extraction = json.load(f)

print(f"\nExtraction data structure:")
print(f"  Type: {type(test_extraction.get('data', 'N/A'))}")
if 'data' in test_extraction and test_extraction['data']:
    if isinstance(test_extraction['data'], list):
        print(f"  Sample: {test_extraction['data'][:2]}")
    elif isinstance(test_extraction['data'], dict):
        items = list(test_extraction['data'].items())[:2]
        for k, v in items:
            print(f"  {k}: {v}")

# Load corresponding ground truth
if config_file.exists():
    with open(config_file, 'r') as f:
        all_configs = json.load(f)
    
    # Find matching ground truth
    matching_gt = None
    for config in all_configs:
        if config.get('id') == chart_id:
            matching_gt = config
            break
    
    if matching_gt:
        print(f"\nGround truth found for {chart_id}:")
        print(f"  Type: {matching_gt.get('type', 'N/A')}")
        
        if 'categories' in matching_gt and 'values' in matching_gt:
            print(f"  Categories: {matching_gt['categories'][:3]}...")
            print(f"  Values: {matching_gt['values'][:3]}...")
        elif 'data' in matching_gt:
            print(f"  Data structure: {type(matching_gt['data'])}")
    else:
        print(f"\n⚠️  No ground truth found for {chart_id}")

# ## 5. Identify the problem

print("\n\n5. PROBLEM IDENTIFICATION:")
print("-" * 40)

# Common issues
issues = []

# Check if extraction data format matches expected format
sample_ext = test_extraction.get('data', {})
if not sample_ext:
    issues.append("Extraction data is empty")
elif isinstance(sample_ext, dict):
    # Check if it's in the right format
    if all(isinstance(v, (int, float, str)) for v in sample_ext.values()):
        issues.append("Extraction uses dict format (key: value)")
    else:
        issues.append("Extraction dict has non-numeric values")
elif isinstance(sample_ext, list):
    if sample_ext and 'category' not in sample_ext[0]:
        issues.append("Extraction list items missing 'category' field")
    if sample_ext and 'value' not in sample_ext[0]:
        issues.append("Extraction list items missing 'value' field")

# Check ground truth format
if matching_gt:
    if 'data' not in matching_gt and 'categories' in matching_gt:
        issues.append("Ground truth uses 'categories'/'values' format, not 'data'")

print("\nIdentified issues:")
for issue in issues:
    print(f"  ❌ {issue}")

# ## 6. Format comparison

print("\n\n6. FORMAT COMPARISON:")
print("-" * 40)

print("\nExtraction format example:")
if test_extraction.get('data'):
    print(f"  {json.dumps(test_extraction['data'], indent=2)[:200]}")

print("\nExpected format for metrics:")
print("  List format: [{'category': 'A', 'value': 10}, {'category': 'B', 'value': 20}]")
print("  OR")
print("  Dict format: {'A': 10, 'B': 20}")

# ## 7. Recommendations

print("\n\n7. RECOMMENDATIONS:")
print("-" * 40)

print("\n1. Check if extraction 'data' field contains actual data")
print("2. Ensure ground truth and extraction formats are compatible")
print("3. Verify category names match between extraction and ground truth")
print("4. Check if perturbation naming follows expected pattern")

# Save diagnostic report
diagnostic_data = {
    'extraction_files_found': len(extraction_files),
    'sample_extraction_structure': {
        'file': test_extraction_file.name,
        'has_data': 'data' in test_extraction,
        'data_type': str(type(test_extraction.get('data', None))),
        'data_empty': not bool(test_extraction.get('data', None))
    },
    'ground_truth_found': matching_gt is not None,
    'identified_issues': issues
}

with open('extraction_diagnostic_report.json', 'w') as f:
    json.dump(diagnostic_data, f, indent=2)

print("\n✅ Diagnostic report saved to: extraction_diagnostic_report.json")
print("\nRun this script to identify why metrics are showing 0%!")
