# PolyCORE lipid causal-analysis code

Minimal public code release for the causal-assumption-guided sweat-to-blood lipid analysis in Figure 5h-i of the PolyCORE lipid sensing manuscript.

This repository is intentionally scoped. It exposes the causal assumptions, adjustment-set rationale, model input definitions, and Figure 5h-i style evaluation code. It does not include wet-lab fabrication protocols, raw electrochemical traces, or restricted participant-level source data.

## What Is Included

- `src/causal_specification.py`: defines the DAG edges, adjustment sets, and causal-assumption-guided prediction feature sets.
- `src/run_causal_adjustment.py`: writes DAG/feature-set tables and computes adjusted blood-to-sweat association diagnostics.
- `src/reproduce_figure5hi.py`: computes 5-fold model metrics and generates Figure 5h-i style plots using the feature definitions from `src/causal_specification.py`.
- `src/make_example_data.py`: creates synthetic schema-compatible data for smoke testing.
- `docs/causal_model.md`: describes the causal graph and how it maps to model inputs.
- `data/merged_data_schema.csv`: expected columns for the restricted paired sweat-blood dataset.
- `data/figure5hi_reported_summary.csv`: aggregate Figure 5h-i reference values extracted from the submitted source-data workbook. This file contains model-level summary statistics only, not participant-level data.
- `source_data/README.md`: explains which source-data workbooks exist and why full workbooks are not included here.

## Data Availability

The paired human-subject dataset is not included in this public code repository. To reproduce the paper-level analysis, place the restricted dataset locally at:

```text
data/merged_data.csv
```

The required schema is documented in:

```text
data/merged_data_schema.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Test With Synthetic Data

This verifies that the pipeline runs. It does not reproduce manuscript values.

```bash
python src/make_example_data.py --out data/example_merged_data.csv
python src/causal_specification.py --out results/causal_specification
python src/run_causal_adjustment.py \
  --data data/example_merged_data.csv \
  --out results/example_causal_adjustment
python src/reproduce_figure5hi.py \
  --data data/example_merged_data.csv \
  --out results/example \
  --bar-source cv \
  --whisker sd
```

## Inspect The Causal Assumptions

The causal assumptions are specified before model fitting:

```bash
python src/causal_specification.py --out results/causal_specification
```

Outputs:

```text
results/causal_specification/
├── causal_dag_edges.csv
└── causal_feature_sets.csv
```

To compute adjusted association diagnostics from a local restricted dataset:

```bash
python src/run_causal_adjustment.py \
  --data data/merged_data.csv \
  --out results/causal_adjustment
```

This writes `causal_adjustment_summary.csv`, where each row reports the residualized blood-to-sweat association after a specified adjustment set. These diagnostics make the adjustment logic auditable; they are separate from the predictive estimator used for Figure 5h-i.

## Reproduce Internally Consistent 5-Fold CV Outputs

Use this mode when bars, error bars, and overlaid points should all come from the same 5-fold evaluation.

```bash
python src/reproduce_figure5hi.py \
  --data data/merged_data.csv \
  --out results/figure5hi_cv \
  --bar-source cv \
  --whisker sd
```

Outputs:

```text
results/figure5hi_cv/
├── figure5hi_fold_metrics.csv
├── figure5hi_summary_metrics.csv
├── figure5hi_bar_tidy.csv
├── figure5hi_points_tidy.csv
├── figure5hi_plot_tidy.csv
├── figure5hi.png
├── figure5h.png
└── figure5i.png
```

## Recreate the Reported Figure 5h-i Reference Bars

The submitted source-data workbook includes aggregate Figure 5h-i values in `data/figure5hi_reported_summary.csv`. To plot those reported bars while overlaying fold-level points computed from the local restricted dataset:

```bash
python src/reproduce_figure5hi.py \
  --data data/merged_data.csv \
  --out results/figure5hi_reported_reference \
  --bar-source reported_summary \
  --reported-summary data/figure5hi_reported_summary.csv \
  --whisker minmax
```

In this mode, output CSVs explicitly mark provenance:

- `bar_source = reported_summary` for bars loaded from the aggregate reference table.
- `whisker_source = 5fold_cv_min_max_enclosing_bar` when min-max whiskers come from fold-level points and are expanded only if needed so the reported bar value is inside the plotted interval.

## Output CSVs

- `figure5hi_fold_metrics.csv`: one row per task, model, and fold.
- `figure5hi_summary_metrics.csv`: model-level means, standard deviations, and min-max values across folds.
- `figure5hi_bar_tidy.csv`: one row per task, model, and metric for plotting bars and whiskers.
- `figure5hi_points_tidy.csv`: one row per task, model, metric, and fold for plotting points.
- `figure5hi_plot_tidy.csv`: merged long-form plotting table.

## Reproducibility Settings

Defaults:

- `KFold(n_splits=5, shuffle=True, random_state=42)`
- Ridge alpha: `1.0`
- Lasso alpha: `0.1`
- Random forest trees: `300`

These can be changed with command-line flags.

## Important Interpretation Notes

The public aggregate summary is suitable for checking reported Figure 5h-i values, but it is not a substitute for the restricted participant-level dataset. The "Causal ML" label here means a predictive estimator whose inputs are selected from the causal graph and adjustment rationale. It should not be interpreted as a complete public release of every exploratory causal script. Repeated measurements from the same participant increase the information available for model fitting; subject-grouped validation should be reported as a sensitivity analysis when evaluating generalization to unseen participants.
