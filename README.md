# FastSpeech 2 assignment — English and Telugu

## Read this first

This is a runnable implementation package, **not a completed empirical submission**.
The supplied inputs contained no datasets or trained checkpoints. Real six-hour data
preparation, alignment and full model training have **not** been performed here.
`Report/Technical_Report.pdf` describes the implementation and explicitly marks
missing experimental evidence. Do not claim that the empty Results/checkpoint folders
are completed results. Synthetic tests are kept separate and deleted after testing.

The pipeline uses the supplied ming024/FastSpeech2 implementation, with its real
FFT encoder/decoder, variance adaptor, length regulator, mel projection and PostNet.
It trains independent weights for the two languages. Full code changes are listed in
`CHANGES.md`; the upstream MIT license is retained.

## 1. Install (Python 3.10–3.12)

Extract the ZIP and open a terminal **inside `project`**. On Windows, PowerShell works;
WSL2/Linux is preferable for GPU training. Internet access is needed for dependencies
and datasets. A CUDA GPU is recommended for full training; CPU is supported but slow.
No fixed runtime or speech quality is promised. Reserve tens of GB of free disk space.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

The PyTorch command above installs the **CPU build**. For NVIDIA training, replace
that command with the CUDA build command matching your system from
https://pytorch.org/get-started/locally/ before installing the other requirements.
Check it with `python -c "import torch; print(torch.cuda.is_available())"`.
If PowerShell activation is restricted, use `.venv\Scripts\python.exe` in place of
`python`; changing system execution policy is unnecessary.

The smoke test checks CTC repeated tokens, split leakage, Telugu encoding, acoustic
features, model training/backpropagation, checkpoint resume, padded durations,
waveform export and plots. It does NOT train useful speech models.

## 2. Download the actual datasets

```bash
python scripts/download_data.py --language English
```

Read the [IndicTTS license](https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf).
If you accept its terms:

```bash
python scripts/download_data.py --language Telugu --accept-indictts-license
```

Sources are precisely the datasets linked by the assignment:
- https://huggingface.co/datasets/ai4bharat/Svarah (`test`, audio column `audio_filepath`)
- https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu (`train`, audio column `audio`)

The downloader exports WAVs and manifests to `data/raw/<Language>/`. Svarah's source
`test` split is re-split only to comply with this TTS assignment; results must not be
represented as a standard Svarah ASR benchmark score. For reproducible future downloads,
pass `--revision <dataset-commit-hash>`; the export records the requested revision and
Hugging Face dataset fingerprint. If access requires authentication, authenticate using
the Hugging Face tooling on your own computer. No token belongs in source code.

Alternatively supply local audio and a UTF-8 JSON list:

```json
[
  {"id":"utt_0001", "audio":"wav/utt_0001.wav", "text":"An actual transcript.", "speaker":"speaker_01"}
]
```

Audio paths are relative to the manifest; IDs must be unique alphanumeric/underscore/
hyphen names. Supply all available recordings, not a manually overlapping split.
For English, real speaker IDs are preferable when available. The HF Svarah schema has
no verified speaker ID, so the default export honestly uses a pooled voice rather than
inventing IDs from demographic fields. Telugu gender identifies the two documented
speakers in this particular corpus. This is not a general gender-to-speaker assumption.

## 3. Preprocess, align, train and synthesize

Run each command only after the preceding command succeeds:

```bash
python English/preprocess.py --manifest data/raw/English/manifest.json
python -m fs2.alignment --config English/config.yaml --device auto
python English/train.py --device auto
python English/inference.py --device auto

python Telugu/preprocess.py --manifest data/raw/Telugu/manifest.json
python -m fs2.alignment --config Telugu/config.yaml --device auto
python Telugu/train.py --device auto
python Telugu/inference.py --device auto

python scripts/build_report.py
python scripts/check_submission.py
```

`auto` selects CUDA if available, otherwise CPU. You may specify `--device cpu` or
`--device cuda`. The scripts default to 40 CTC alignment epochs and 150 FastSpeech 2
epochs; these are starting settings, not evidence of convergence. Change the separate
YAML files as needed **before preprocessing/alignment**. Smaller batch sizes reduce
GPU memory use. Tensorboard is not needed: JSON history and PNG curves are saved each epoch.

A convenience runner performs the same commands (including download) for one language:

```bash
python scripts/run_pipeline.py --language English --device auto
python scripts/run_pipeline.py --language Telugu --device auto --accept-indictts-license
```

Use either the staged commands OR the runner, not both into the same prepared directory.
For a local manifest, pass `--manifest /absolute/path/manifest.json` to the runner.
Preprocessing refuses to overwrite prepared data. To start a genuinely new experiment,
change `data_dir`, `results_dir` and `checkpoint_dir` in a copied config, then use
`--config <that-file>` on the staged commands. Existing files are never silently deleted.

Resume an interrupted FastSpeech 2 run at the last completed epoch:

```bash
python English/train.py --resume English/checkpoints/last.pt --epochs 150
python Telugu/train.py --resume Telugu/checkpoints/last.pt --epochs 150
```

`--epochs` is the total target epoch, not additional epochs. The best validation checkpoint
is used for inference; optimizer and RNG states are in `last.pt`. Load only checkpoints
you trust. Checkpoints include configuration, feature statistics and dataset identity.
Do not change architecture or preprocessing when resuming. The CTC alignment stage
currently restarts rather than resumes; `--epochs 80` trains a longer aligner from scratch.

## 4. Expected outputs after real runs

- `<Language>/checkpoints/best.pt` and `last.pt`
- `Results/<Language>/training_loss.png`, `validation_loss.png`, `loss_history.json`
- `Results/<Language>/test_generated_audios/{E1,E2a,E2b,E3,E4}.wav` or Telugu IDs
- Per-sentence `.mel.npy` and `.json` (original text, model text, units, predictions)
- `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`
- `data/<Language>/metadata.json` (actual selected hours/IDs/hashes), `rejected.json`,
  `alignment_training.json`, `alignment_diagnostics.json`, `stats.json`, vocab and speakers
- `Report/Technical_Report.pdf`, regenerated from available real outputs

The paired a/b sentences count separately: five English and five Telugu utterances.
The deliberately broken `rhyth m` and `అడ విలో` are preserved. Sentence files use a visual
transcription of the PDF; PDF extraction introduces spurious gaps around Telugu marks.
The original strings are retained in output JSON. NFC normalization and lowercasing are
recorded separately; no automatic spelling correction or date/time expansion is used.

By default inference chooses the training speaker/group with most selected audio. To
choose another, inspect `data/<Language>/speakers.json` and pass `--speaker <key>`.
For custom text, create a JSON list of `{id,text}` objects and pass `--sentences file.json`.

## 5. Method and limitations

- Seed 42; duplicate decoded recordings removed; normalized identical transcripts kept
  in the **same split**, including the two Telugu voices. Training is approximately 5 h,
  validation 1 h, with at most 60 seconds excess per split. Actual totals are checked.
  Speaker overlap is permitted: validation measures held-out text for known voices,
  not zero-shot speakers. Unsupported transcripts and duration filtering are logged.
- 22,050 Hz mono; 1024-point STFT; hop 256; 80 natural-log power mel bands, 0–8000 Hz.
  pYIN estimates F0 (50–600 Hz); unvoiced frames are zero. Energy is RMS amplitude.
- To make both languages trainable without an unavailable Telugu forced-alignment model,
  an independent train-only two-layer bidirectional LSTM CTC aligner supplies code-point
  durations via constrained Viterbi. These are **character durations, not phoneme durations**.
  Blank frames attach to the preceding token; leading blanks attach to the first.
  A train-subset greedy CER gate (<=0.35) prevents obviously collapsed alignment exports.
  This gate is necessary but insufficient: inspect per-utterance CER and duration outliers
  before relying on the targets. Validation does not fit or select this aligner.
- Pitch is the mean of voiced frames within each token, or zero if wholly unvoiced;
  energy is the token mean RMS. Train-only means/stds normalize both; validation uses them
  unchanged. Zero-F0 tokens are included in the normalization and loss.
- Upstream token-level embeddings condition the model on ground-truth variances in training
  and predictions in inference. Loss: mel L1 + postnet L1 + duration log(1+d) MSE + pitch
  MSE + energy MSE, with padding masked. Validation is teacher-forced, so free-running
  speech must be separately inspected. Epoch values average batch means by batch size.
- No neural vocoder checkpoint was supplied. A consistent power-mel inversion and
  64-iteration Griffin–Lim provide a fully local baseline, with limited naturalness.
  It is not HiFi-GAN and not the paper's waveform model. For stronger results, train a
  matching neural vocoder; arbitrary pretrained vocoders can expect incompatible mels.
- English pooled speaker variability, Telugu combining-character durations, punctuation
  alignment and unseen numerals/code-switching are known limitations. Do not expect the
  model to reliably solve heteronyms, semantic ambiguity or time normalization unaided.

## 6. Complete the report and submit

After listening, fill `Report/listening_notes.json`: pronunciation, quality, duration,
pitch and energy observations for all 10 sentences. Use the saved token index mappings
and compare E2a/E2b and T2a/T2b. Add measured observations, not expected behavior.
Regenerate the PDF. It automatically includes measured hours, loss summaries, curves
and variance statistics where they exist, and notes missing evidence otherwise.
`check_submission.py` fails while checkpoints, outputs or listening notes are absent.

Create your own public GitHub repository and upload this project after finishing the
experiments. This ZIP does not create a repository or submit the assignment. For large
checkpoints use Git LFS or durable accessible download links, as the assignment allows.
Retain the source manifests, split files, vocabularies, statistics and config alongside
checkpoints for reproducibility; raw datasets need not be published. Follow dataset terms.

## Troubleshooting

- **Insufficient hours:** read `rejected.json`; verify complete datasets, supported text,
  and duplicate groups. Do not silently reduce required hours or reuse validation audio.
- **Alignment CER gate failed:** increase aligner epochs, inspect transcripts/recordings,
  and compare the aligner log. Do not disable the gate just to obtain files.
- **Zero/excessive duration:** inference fails with a clear message; inspect convergence
  and alignment. It does not fabricate durations to make an untrained model speak.
- **Out of memory:** reduce batch size before the experiment; use shorter maximum audio
  if enough hours remain, and regenerate the affected data. Keep sequence lengths within
  `max_seq_len` (default 3000).
- **Import errors:** use the same virtual environment for installation and commands.
- **Telugu plot fonts:** plots use numeric token indices with full Unicode mappings in
  JSON, so missing Telugu fonts cannot corrupt the quantitative curves.
