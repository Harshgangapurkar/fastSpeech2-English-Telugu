# Verification record

Executed in the authoring environment on 7 November 2026, CPU only.
Python: 3.12.14

- torch: 2.15.0+cpu
- numpy: 2.3.5
- scipy: 1.17.0
- librosa: 0.11.0
- soundfile: 0.13.1
- matplotlib: 3.10.8
- PyYAML: 6.0.3
- reportlab: 4.4.9

`python scripts/smoke_test.py`: 7 automated tests passed.
These verify CTC repeated-token alignment and full frame coverage, train-only feature
statistics, Unicode preservation, duplicate protection and deterministic splits,
real waveform feature extraction, CTC backpropagation, and FastSpeech 2 train/resume/
waveform/plot export for both language paths. Temporary fixtures use tiny architectures
and synthetic audio/features. They prove execution mechanics, not speech quality or
convergence. Temporary checkpoints, synthetic WAVs and fixture plots are deleted.
The synthetic alignment export test alone relaxes its quality gate; real configs retain
CER <= 0.35. The export-mechanics test fixes duration biases only in temporary weights.

`python -m compileall -q .`: all Python source parsed successfully.
Entry-point `--help` checks passed for both languages and the shared commands.
`python scripts/check_submission.py`: intentionally exits 1, correctly identifying
missing real checkpoints, results and listening observations.
The generated technical report was rendered and visually inspected.

Not tested here: multi-gigabyte dataset download, six-hour preprocessing runs, full-size
GPU training, full-corpus alignment convergence, CUDA execution, Windows execution,
real sentence intelligibility, neural vocoding or GitHub publication. Dependency ranges
are portability choices; this record is the actual tested environment. The optional
Hugging Face exporter uses datasets 3.6.0's documented audio-array interface.
