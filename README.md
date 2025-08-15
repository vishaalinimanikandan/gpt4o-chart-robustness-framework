# GPT-4o Chart Robustness Framework

A comprehensive evaluation framework for assessing the robustness of GPT-4o in extracting structured data from charts under various visual perturbations.

## Overview

This repository contains the complete implementation of a systematic evaluation framework designed to assess how well GPT-4o can extract structured data from charts when subjected to real-world visual distortions. The framework generates synthetic charts, applies controlled perturbations, and evaluates extraction performance using multiple metrics.

## Research Objectives

- **Benchmark Creation**: Generate 1,400 synthetic charts across 5 chart types and 3 complexity levels
- **Robustness Testing**: Apply 6 types of visual perturbations simulating real-world conditions
- **Zero-Shot Evaluation**: Test GPT-4o's chart understanding without task-specific training
- **Multi-Metric Assessment**: Evaluate performance using Value Accuracy, F1 Score, Composite Score, and Relative Accuracy

## Framework Architecture

```
Dataset Generation → Perturbation Engine → GPT-4o Extraction → Evaluation Engine
```

### Chart Types Supported
- **Bar Charts**: Categorical data comparison
- **Line Charts**: Trend analysis and time series
- **Pie Charts**: Composition and percentage data
- **Scatter Plots**: Correlation and distribution analysis
- **Area Charts**: Cumulative data visualization

### Complexity Levels
- **Medium**: 6-10 data points, 2-3 series, basic annotations
- **Complex**: 8-12 data points, 3-4 series, dual axes or error bars
- **Advanced**: 10-15 data points, 4-6 series, annotations, dual axes, error bars

### Perturbation Types
- **Gaussian Blur**: σ=1.0 (lossy compression simulation)
- **Rotation**: ±5° (document misalignment)
- **Brightness Shift**: -40% (poor lighting conditions)
- **Grayscale Conversion**: Color information removal
- **Random Block Occlusion**: 10% area coverage (stamps, watermarks)
- **Legend Corruption**: Partial legend removal (cropping damage)

## Key Findings

- **Robust to minor distortions**: GPT-4o maintains 50+ Composite Scores under blur, grayscale, and rotation
- **Sensitive to occlusion**: Block occlusion causes 20-30% performance degradation
- **Chart type variations**: Line and scatter charts show highest robustness
- **Complexity impact**: Performance decreases with structural complexity

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages (see `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/vishaalinimanikandan/gpt4o-chart-robustness-framework.git
cd gpt4o-chart-robustness-framework

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### Usage

1. **Generate Charts** (Notebook 02):
```bash
jupyter notebook 02_Advanced_Chart_Generation.ipynb
```

2. **Apply Perturbations** (Notebook 03):
```bash
jupyter notebook 03_Perturbation_Framework.ipynb
```

3. **Run GPT-4o Extraction** (Notebook 04):
```bash
jupyter notebook 04_GPT4_Extraction_Pipeline.ipynb
```

4. **Analyze Results** (Notebook 05):
```bash
jupyter notebook 05_Comprehensive_Analysis.ipynb
```

## Repository Structure

```
├── 02_Advanced_Chart_Generation.ipynb    # Chart dataset generation
├── 03_Perturbation_Framework.ipynb       # Visual perturbation engine
├── 04_GPT4_Extraction_Pipeline.ipynb     # GPT-4o extraction pipeline
├── 05_Comprehensive_Analysis.ipynb       # Results analysis and visualization
├── Data_analysis.ipynb                   # Additional data analysis
├── research_config.json                  # Framework configuration
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # This file
```

## Evaluation Metrics

### Value Accuracy (VA)
Percentage of numeric values extracted within ±10% tolerance of ground truth.

### F1 Score
Structural matching of predicted vs. ground truth category-value pairs.

### Composite Score (CS)
Combined measure: `(F1 Score + Value Accuracy) / 2 × 100`

### Relative Accuracy (RA)
Performance retention: `(Perturbed Score / Clean Score) × 100`

## Research Contributions

1. **First systematic robustness evaluation** of VLMs for chart extraction
2. **Synthetic benchmark dataset** with controlled complexity and perturbations
3. **Multi-dimensional analysis** across chart types, domains, and complexity levels
4. **Reproducible framework** for evaluating other VLMs
5. **Practical insights** for real-world deployment considerations

## Sample Results

| Perturbation Type | Mean Composite Score | 95% CI | Relative Accuracy |
|-------------------|---------------------|---------|-------------------|
| Original (Clean)  | 51.22              | 49.1-53.3 | 100.0% |
| Blur              | 51.52              | 49.4-53.6 | 100.6% |
| Rotation          | 50.97              | 48.7-53.2 | 99.5% |
| Grayscale         | 48.38              | 45.9-50.9 | 94.5% |
| Block Occlusion   | 39.85              | 37.1-42.6 | 77.8% |

## Configuration

The framework is highly configurable through `research_config.json`:

```json
{
  "chart_generation": {
    "total_charts": 200,
    "chart_types": ["bar", "line", "pie", "scatter", "area"],
    "complexity_distribution": {"medium": 0.4, "complex": 0.4, "advanced": 0.2}
  },
  "perturbations": {
    "types": ["blur", "rotation", "brightness", "grayscale", "blocks", "corruption"],
    "intensity": "medium"
  },
  "evaluation": {
    "metrics": ["value_accuracy", "f1_score", "composite_score", "relative_accuracy"]
  }
}
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{manikandan2025robustness,
  title={Robustness Evaluation of GPT-4o for Structured Data Extraction from Charts},
  author={Manikandan, Vishaalini Ramasamy},
  year={2025},
  school={Trinity College Dublin},
  type={Master's Thesis}
}
```

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest improvements
- Add support for new chart types
- Implement additional perturbation types
- Extend evaluation metrics



For questions about this research or framework, please open an issue in this repository.

---

**Keywords**: Chart Data Extraction, Vision-Language Models, GPT-4o, Robustness Evaluation, Visual Perturbations, Structured Data, Synthetic Benchmark, Zero-Shot Learning, Data Accuracy Metrics, Multimodal AI
