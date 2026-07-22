# Assignment 5 Student Examples

These examples are small helper patterns for Assignment 5. They are not complete solutions.

Use them to produce evidence for:

- split validity,
- learning curves,
- error or slice analysis,
- confidence or calibration discussion.

## Files

- `split_audit_examples.py`: split counts, feature/target distributions by split, and category coverage checks.
- `plot_learning_curves_example.py`: plot training and validation curves from one or more history CSV files.
- `error_slice_and_confidence_examples.py`: regression slice errors, classification confusion/per-class summaries, and confidence summaries.

## Expected Inputs

These examples assume you already have CSV or Parquet artifacts from your own run. Adapt column names to match your project.

For split audits, common columns are:

- `split`
- target or label column, such as `target_duration_seconds` or `label`
- input columns worth checking across splits, such as `trip_distance`, `pickup_hour`, `source`, or `patient_id`

For learning curves, common columns are:

- `epoch`
- `train_loss`
- `validation_loss`
- a validation metric such as `validation_mae`, `validation_accuracy`, or `validation_macro_f1`

For error/confidence analysis, common columns are:

- regression: `y_true`, `y_pred`, and useful slice columns
- classification: `true_label`, `predicted_label`, optional `confidence`, and useful slice columns
