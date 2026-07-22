# Split Audit Summary (Beatport Key Task)

## Song-level counts (`key_splits.csv`)

| split | rows | percent |
| --- | --- | --- |
| train | 785 | 69.96434937611407 |
| val | 168 | 14.973262032085561 |
| test | 169 | 15.062388591800357 |

## Chunk-level counts (`clip_15s/key_chunks.csv`)

| split | rows | percent |
| --- | --- | --- |
| train | 6280 | 69.96434937611407 |
| val | 1344 | 14.973262032085561 |
| test | 1352 | 15.062388591800357 |

## Leakage checks

| check | passed | violation_count | sample_track_ids |
| --- | --- | --- | --- |
| track_in_multiple_splits | True | 0 |  |

| check | passed | violation_count | sample_track_ids |
| --- | --- | --- | --- |
| chunk_split_consistency | True | 0 |  |

## Label coverage vs train (val/test keys unseen in train?)

| column | split | train_unique_values | split_unique_values | unseen_values_count | unseen_values_sample | unseen_row_count | unseen_row_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| key_24 | val | 24 | 24 | 0 |  | 0 | 0.0 |
| key_24 | test | 24 | 24 | 0 |  | 0 | 0.0 |
| mode | val | 2 | 2 | 0 |  | 0 | 0.0 |
| mode | test | 2 | 2 | 0 |  | 0 | 0.0 |
| pitch_class | val | 12 | 12 | 0 |  | 0 | 0.0 |
| pitch_class | test | 12 | 12 | 0 |  | 0 | 0.0 |
