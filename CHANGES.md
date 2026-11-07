# Source provenance and modifications

Starting archive: user-supplied FastSpeech2-master.zip; archive comment identifies commit
`d4e79eb52e8b01d24703b2dfc0385544092958f3` of https://github.com/ming024/FastSpeech2.
`LICENSE.upstream` retains the MIT license. `UPSTREAM_README.md` is the original README.
Original demo WAVs and external pretrained models are not assignment results and are omitted.

## Modified upstream files under fs2/

- `model/fastspeech2.py`: package-relative imports to the adapted modules/utilities.
- `model/modules.py`: package imports; length tensors follow the input device instead
  of global CUDA availability; token energy uses `e_control` (upstream used `p_control`);
  multiply duration control before rounding; mask padded durations; fail on zero-length
  or excessive predicted sequences; clamp log duration before exponentiation.
- `transformer/Models.py`: configured vocabulary size replaces the English symbol list;
  packaged Constants import. Other transformer layers retain upstream implementations.
- `model/__init__.py`: only exports the model used by this pipeline. Original loss and
  optimizer source are retained for reference but the pipeline uses its documented
  masked losses and AdamW in `fs2/train.py`.
- `tensor_utils.py`: minimal padding/masking utilities avoid legacy unrelated imports.

## New shared pipeline

Common UTF-8 code-point frontend, corpus exporter, data audit and transcript-group splits,
features, CTC aligner, masked training/resume, inference, variance export, report builder,
submission checker and automated contract tests. The original MFA/TextGrid preprocessing
and English G2P are replaced rather than falsely presented as Telugu-compatible.

English and Telugu wrappers share implementation to avoid divergence but use separate
configs, inventories, training sets, alignment weights, FS2 weights and output folders.
English has a pooled voice by default because verified speaker IDs are absent from the
provided HF schema. Telugu uses its documented two voices, preserves dependent vowels,
virama and code points, and avoids ASCII conversion. Neither configuration uses the other
language's trained checkpoint. ASCII is retained in Telugu for numbers and mixed text.

## Differences from the research paper

This is a character-input FastSpeech 2 adaptation of the supplied implementation, not a
claim of paper-exact reproduction. It uses token-averaged F0 rather than CWT pitch,
RMS energy rather than a spectral L2 target, a learned CTC duration teacher rather than
MFA phoneme alignment, AdamW at constant LR, and Griffin-Lim instead of a neural vocoder.
These decisions make an independent bilingual baseline runnable but limit quality.
