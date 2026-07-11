# Data

`merged_data.csv` is not included because it contains restricted human-subject data.

To run the full analysis locally, place the restricted paired sweat-blood dataset here:

```text
data/merged_data.csv
```

For a smoke test, generate synthetic data:

```bash
python src/make_example_data.py --out data/example_merged_data.csv
python src/reproduce_figure5hi.py --data data/example_merged_data.csv --out results/example
```

