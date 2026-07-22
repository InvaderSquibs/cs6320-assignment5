# Data validation artifacts

## Splits
See `split_summary.csv` — chunk/segment/track counts per train/val/test.

## Listen manually
- `listen_samples.csv` — paths and labels for 3 clips per split
- `listen_samples.html` — open in a browser to play MP3 segments

## Visual audio check
`spectrograms/` — mel + chroma plots for 6 random chunks (QA only)

## PyTorch input check
`pytorch_sample_records.json` — for each sample:
- row metadata (split, key, segment, time range)
- cached stereo waveform stats (2 x 661500 float32 PCM @ 44100 Hz)
- label class index and tensor shape the CNN sees

This matches `assignment_5/train_key_cnn.py`.
