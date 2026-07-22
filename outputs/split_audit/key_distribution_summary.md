# Key distribution by split

Clip length: **15s**

## Segment-level (split unit for modeling)

- Total segments: 1143 ({'train': 800, 'test': 172, 'val': 171})
- Unique keys in train: 24
- Timed segments: 35

See `segment_key_counts_by_split.csv` and PNG plots in this folder.

## Chunk-level (what the classifier trains on)

- Total chunks: 8976

Use **share curves** to verify stratification: train/val/test lines should follow similar shapes.
