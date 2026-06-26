# LogACF: An Agent Collaborative Framework for Well Logging Data Processing and Interpretation

## Overview

**LogACF** is a large language model-based agent collaborative framework for well logging data processing, reservoir property prediction, reservoir zone identification, visualization, and interpretation report generation.

This project is associated with the paper:

**Large language model-based AI agent for well logging data processing and interpretation**
Authors: Gang Luo, Li-Zhi Xiao
Journal: *Petroleum Science*
DOI: `10.1016/j.petsci.2026.05.031`

The framework integrates large language models, Python-based data analysis, well log visualization, reservoir evaluation rules, multimodal image interpretation, and Markdown-to-PDF report generation into a unified workflow.

The main workflow includes:

1. Well logging task understanding and decomposition.
2. AI-assisted Python code generation and iterative correction.
3. Reservoir property prediction.
4. Standardized prediction result export.
5. Model evaluation using RMSE.
6. Reservoir zone identification and quality classification.
7. Well log visualization.
8. Vision-based interpretation of generated well log figures.
9. Professional Markdown report generation.
10. PDF report export.

---

## Project Source

This repository provides an implementation and engineering demonstration of the method proposed in:

```text
Large language model-based AI agent for well logging data processing and interpretation
Gang Luo, Li-Zhi Xiao
Petroleum Science
DOI: 10.1016/j.petsci.2026.05.031
```

The project implements the core idea of an AI-agent-based well logging interpretation framework. It demonstrates how large language models can coordinate multiple specialized agents to complete data processing, code generation, model evaluation, reservoir interpretation, visualization, and report generation tasks.

---

## Demo Data Source

The demonstration data used in this project are derived from the **SPWLA Machine Learning Competition 2021** public dataset:

```text
SPWLA Machine-Learning-Competition-2021-main
```

Dataset repository:

```text
https://github.com/pddasig/Machine-Learning-Competition-2021
```

The dataset is used in this project to demonstrate the complete LogACF workflow, including well logging data processing, reservoir property prediction, model evaluation, reservoir zone identification, and interpretation report generation.

If you use the demo data or reproduce the related experiments, please also cite:

```text
Fu, L., Yu, Y., Xu, C., Ashby, M., McDonald, A., Pan, W., ... & Lee, J. (2024).
Well-Log-Based Reservoir Property Estimation With Machine Learning: A Contest Summary.
Petrophysics, 65(01), 108-127.
DOI: 10.30632/PJV65N1-2024a6
```

---

## Project Structure

```text
LogACF/
│
├── LogACF.py
├── plot_well_log_data.py
├── plot_well_predictions.py
├── convert_PDF.py
│
├── test.csv
├── real_test_result.csv
│
├── documents/
│   ├── Data Description.txt
│   ├── Device Information.txt
│   ├── Knowledge.txt
│   ├── Pictures.txt
│   ├── Python Packages.txt
│   ├── Report.txt
│   ├── Task Requirement.txt
│   └── Tasks Description.txt
│
│
├── Results/
│   ├── <timestamp>/
│   │   └── model_predictions.csv
│   │
│   └── Pictures/
│       ├── Well_<well_number>.png
│       ├── Well_<well_number>.pdf
│       ├── Well_<well_number>_comparison.png
│       ├── Well_<well_number>_comparison.pdf
│       ├── Well_<well_number>_reservoir_zones.csv
│       ├── Well_<well_number>_reservoir_report.txt
│       └── overall_analysis.json
│
└── README.md
```

---

## Core Modules

### 1. `LogACF.py`

`LogACF.py` is the main orchestration script of the framework.

It coordinates agent and connects the following tasks:

* Task decomposition
* Web research assistance
* Python code generation
* Code execution and error correction
* Model prediction result validation
* Best result selection
* Reservoir analysis
* Well log plotting
* Vision-based interpretation
* Markdown report generation
* PDF report conversion

The script defines several agent roles:

| Agent                         | Role                                                 |
| ----------------------------- | ---------------------------------------------------- |
| `Well_logging_analysis`       | Main well logging analyst and task coordinator       |
| `Web_research_assistant`      | Internet search and background information assistant |
| `Code_programming_assistant`  | Python code generation and improvement assistant     |
| `Report_generation_assistant` | Professional well logging report writing assistant   |

The generated model prediction file must be named:

```text
model_predictions.csv
```

and must include the following required columns:

```text
WELLNUM
DEPTH
PHIF_pred
SW_pred
VSH_pred
```

---

### 2. `plot_well_log_data.py`

`plot_well_log_data.py` is used to visualize original well logging curves.

It produces a three-track well log figure:

| Track   | Curves          |
| ------- | --------------- |
| Track 1 | GR + CALI       |
| Track 2 | RDEP + RMED     |
| Track 3 | DTC + NEU + DEN |

Main functions include:

* Loading well log data.
* Handling missing values.
* Checking whether each curve is plottable.
* Identifying reservoir zones using the GR cutoff method.
* Drawing reservoir intervals on well log figures.
* Adding depth ticks and depth markers.
* Exporting PNG and PDF well log figures.
* Saving reservoir identification reports.

Typical outputs include:

```text
Results/Pictures/Well_<well_number>.png
Results/Pictures/Well_<well_number>.pdf
Results/reservoir_analysis_report.txt
```

---

### 3. `plot_well_predictions.py`

`plot_well_predictions.py` performs prediction-based reservoir interpretation.

It uses model prediction results and reference data to evaluate model performance and identify reservoir zones.

Main functions include:

* Loading `model_predictions.csv`.
* Loading `real_test_result.csv`.
* Merging prediction and reference data by `WELLNUM` and `DEPTH`.
* Calculating RMSE for PHIF, SW, and VSH.
* Identifying reservoir zones using petrophysical thresholds.
* Classifying reservoir quality.
* Calculating reservoir properties.
* Generating comparison figures.
* Exporting reservoir zone data and reports.

Typical outputs include:

```text
Results/Pictures/Well_<well_number>_comparison.png
Results/Pictures/Well_<well_number>_comparison.pdf
Results/Pictures/Well_<well_number>_reservoir_zones.csv
Results/Pictures/Well_<well_number>_reservoir_report.txt
Results/Pictures/overall_analysis.json
```

---

### 4. `convert_PDF.py`

`convert_PDF.py` converts Markdown reports into PDF files.

Main functions include:

* Reading Markdown files.
* Converting Markdown to HTML.
* Supporting Markdown table rendering.
* Applying CSS styles.
* Calling `wkhtmltopdf` through `pdfkit`.
* Exporting PDF reports.
* Cleaning temporary HTML files.

Typical usage:

```python
from convert_PDF import convert_markdown_to_pdf

convert_markdown_to_pdf(
    input_md_path="final_enhanced_well_logging_report_with_summary.md",
    output_pdf_path="Results/final_report.pdf"
)
```

---

## Data Requirements

### Raw Well Log Data

The raw well log data file is usually named:

```text
test.csv
```

It should contain the following columns:

```text
WELLNUM
DEPTH
GR
CALI
RDEP
RMED
DTC
NEU
DEN
```

Column descriptions:

| Column    | Description            |
| --------- | ---------------------- |
| `WELLNUM` | Well number            |
| `DEPTH`   | Depth                  |
| `GR`      | Gamma ray              |
| `CALI`    | Caliper                |
| `RDEP`    | Deep resistivity       |
| `RMED`    | Medium resistivity     |
| `DTC`     | Compressional slowness |
| `NEU`     | Neutron porosity       |
| `DEN`     | Bulk density           |

Missing values are usually represented by:

```text
-9999
```

---

### Reference Test Data

The reference test data file is usually named:

```text
real_test_result.csv
```

It should contain:

```text
WELLNUM
DEPTH
PHIF
SW
VSH
```

Column descriptions:

| Column | Description                |
| ------ | -------------------------- |
| `PHIF` | Reference porosity         |
| `SW`   | Reference water saturation |
| `VSH`  | Reference shale volume     |

---

### Model Prediction Data

The model prediction file must be named:

```text
model_predictions.csv
```

It should contain:

```text
WELLNUM
DEPTH
PHIF_pred
SW_pred
VSH_pred
```

Recommended output path:

```text
Results/<timestamp>/model_predictions.csv
```

Column descriptions:

| Column      | Description                |
| ----------- | -------------------------- |
| `PHIF_pred` | Predicted porosity         |
| `SW_pred`   | Predicted water saturation |
| `VSH_pred`  | Predicted shale volume     |

---

## Usage

### 1. Run the Full Workflow

```bash
python LogACF.py
```

This runs the full agent workflow, including:

1. Task decomposition.
2. Code generation.
3. Code execution.
4. Prediction result validation.
5. Reservoir analysis.
6. Well log plotting.
7. Vision-based interpretation.
8. Report generation.
9. PDF conversion.

---

## Reservoir Identification Methods

The project supports two reservoir identification methods.

---

### 1. GR-Based Reservoir Identification

Implemented in:

```text
plot_well_log_data.py
```

Reservoir intervals are identified when:

```text
GR < gr_cutoff
```

Default parameters:

```python
gr_cutoff = 60
min_thickness = 5
gap_threshold = 10
```

This method is mainly used for quick visualization based on original well logs.

---

### 2. Prediction-Based Reservoir Identification

Implemented in:

```text
plot_well_predictions.py
```

For each reservoir zone, the following properties are calculated:

* Thickness
* Average PHIF
* Average SW
* Average SO
* Average VSH
* Net-to-gross ratio
* Effective reservoir ratio
* Oil-bearing interval ratio
* Average sampling interval
* Maximum depth gap

---

## Model Evaluation

The framework evaluates prediction performance using RMSE.

Evaluated targets:

```text
PHIF
SW
VSH
```

The average RMSE is calculated as:

```text
Average RMSE = (PHIF_RMSE + SW_RMSE + VSH_RMSE) / 3
```

The framework can search prediction outputs in the `Results` directory and select the best result based on RMSE.

---

## Output Files

After running the workflow, the project may generate:

```text
Results/
│
├── <timestamp>/
│   └── model_predictions.csv
│
├── Pictures/
│   ├── Well_<well_number>.png
│   ├── Well_<well_number>.pdf
│   ├── Well_<well_number>_comparison.png
│   ├── Well_<well_number>_comparison.pdf
│   ├── Well_<well_number>_reservoir_zones.csv
│   ├── Well_<well_number>_reservoir_report.txt
│   └── overall_analysis.json
│
└── final_enhanced_well_logging_report_with_summary.md
```

---

## Report Generation

The final report integrates:

* Task description
* Data description
* Well logging domain knowledge
* Prediction results
* Model evaluation results
* Reservoir analysis results
* Generated well log figures
* Vision-based interpretation
* Executive summary

The report is first generated in Markdown format and can then be converted into PDF.

Recommended input files:

```text
documents/Data Description.txt
documents/Device Information.txt
documents/Knowledge.txt
documents/Pictures.txt
documents/Python Packages.txt
documents/Report.txt
documents/Task Requirement.txt
documents/Tasks Description.txt
```

---

## Recommended Workflow

```text
Step 1: Prepare test.csv and real_test_result.csv
Step 2: Configure API keys
Step 3: Run LogACF.py
Step 4: Generate model_predictions.csv
Step 5: Evaluate prediction results
Step 6: Identify reservoir zones
Step 7: Generate well log figures
Step 8: Generate the Markdown report
Step 9: Convert the Markdown report to PDF
```

---

## Notes

1. Make sure `WELLNUM` and `DEPTH` are consistent across all CSV files.
2. `model_predictions.csv` must contain `PHIF_pred`, `SW_pred`, and `VSH_pred`.
3. `wkhtmltopdf` must be installed before using PDF conversion.
4. Some paths in the scripts are Windows-style paths. Modify them if running on Linux or macOS.
5. Remove all hard-coded API keys before publishing or sharing the project.
6. Generated figures are saved in both PNG and PDF formats.
7. PDF outputs are suitable for reports, slides, and further editing.
8. The demo data are derived from the SPWLA Machine Learning Competition 2021 dataset.
9. The project framework is associated with the LogACF paper published in *Petroleum Science*.

---

## Citation

If you use this project, please cite the following paper:

```bibtex
@article{Luo2026LogACF,
  title   = {Large language model-based AI agent for well logging data processing and interpretation},
  author  = {Luo, Gang and Xiao, Li-Zhi},
  journal = {Petroleum Science},
  year    = {2026},
  doi     = {10.1016/j.petsci.2026.05.031}
}
```

If you use the demonstration data from SPWLA Machine Learning Competition 2021, please also cite:

```bibtex
@article{Fu2024WellLogContest,
  title   = {Well-Log-Based Reservoir Property Estimation With Machine Learning: A Contest Summary},
  author  = {Fu, L. and Yu, Y. and Xu, C. and Ashby, M. and McDonald, A. and Pan, W. and others and Lee, J.},
  journal = {Petrophysics},
  volume  = {65},
  number  = {01},
  pages   = {108--127},
  year    = {2024},
  doi     = {10.30632/PJV65N1-2024a6}
}
```

---

## Reproducibility Statement

Due to the stochastic nature of large language models, results may vary between runs. Therefore, this repository does not guarantee exact reproduction of the case studies, generated code, figures, or reports presented in the associated paper.
