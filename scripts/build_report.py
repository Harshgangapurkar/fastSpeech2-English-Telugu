"""Generate a transparent technical report from implementation and actual run artifacts."""
from pathlib import Path
import sys
import json
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fs2.common import ROOT, read_json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
font_dir=Path(reportlab.__file__).parent/'fonts'
pdfmetrics.registerFont(TTFont('ReportSans',str(font_dir/'Vera.ttf')))
pdfmetrics.registerFont(TTFont('ReportSansBold',str(font_dir/'VeraBd.ttf')))
styles=getSampleStyleSheet()
styles['Title'].fontName='ReportSansBold'
styles['Title'].fontSize=22
styles['Heading1'].fontName='ReportSansBold'
styles['Heading1'].fontSize=14
styles.add(ParagraphStyle(name='BodyCustom',fontName='ReportSans',fontSize=9,leading=12.5,spaceAfter=9))
styles['Title'].textColor=colors.HexColor('#15354b');styles['Heading1'].textColor=colors.HexColor('#15354b')
story=[]
def para(text):story.append(Paragraph(escape(text),styles['BodyCustom']))
def heading(text):story.append(Paragraph(escape(text),styles['Heading1']))
def table(rows,widths):
    data=[[Paragraph(escape(str(x)),styles['BodyCustom']) for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5eef3')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#b7c6ce')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,12))
def page(title):story.append(PageBreak());heading(title)
def maybe(path):return read_json(path) if path.exists() else None

def build():
    story.clear()
    story.append(Paragraph('FastSpeech 2<br/>English and Telugu',styles['Title']))
    para('Technical implementation report | Assignment dated 7 November 2026')
    heading('Execution status and evidence')
    para('This package adapts the supplied FastSpeech2-master source. The delivered package contains implementation and synthetic contract tests. The actual Svarah and IndicTTS training runs were not executed in the authoring environment. There are no supplied trained language checkpoints, assignment speech samples, or measured convergence/quality claims. This report is therefore an implementation report with an explicit experimental completion plan, not a finished empirical report.')
    rows=[['Deliverable','Current artifact evidence']]
    for lang in ['English','Telugu']:
        results=ROOT/'Results'/lang
        rows.append([lang+' training', 'Loss history available' if (results/'loss_history.json').exists() else 'NOT RUN: checkpoint and measured loss curves absent'])
        rows.append([lang+' synthesis','Inference export available; listening still requires notes' if (results/'inference_summary.json').exists() else 'NOT RUN: five test outputs and variance plots absent'])
    table(rows,[135,365])
    para('Regenerate this PDF after running both pipelines and adding listening notes. The generated measurements on later pages reflect files present at report-generation time. The status of the originally delivered package above remains part of its provenance.')
    heading('Architecture and objective')
    para('Unicode token IDs enter the upstream FFT encoder. Speaker/group embeddings condition the encoder states. Duration, pitch and energy predictors produce token-level values; pitch and energy embeddings condition the states and a length regulator expands them to mel frames. The upstream FFT decoder, linear mel projection and five-layer convolutional PostNet predict 80-band log-power mel spectra. Griffin-Lim reconstructs audio from these predicted spectra.')
    para('The training objective is mel L1 + postnet mel L1 + duration MSE on log(1 + frame count) + normalized pitch MSE + normalized energy MSE. All losses mask padded positions. Ground-truth variances condition training and validation; free-running inference uses predicted variances. Validation loss alone cannot establish intelligibility or pronunciation quality.')
    page('Implementation and language adaptations')
    heading('Code provenance')
    para('Reference: ming024/FastSpeech2, supplied archive commit d4e79eb52e8b01d24703b2dfc0385544092958f3. The MIT license and upstream README are retained. The FFT layers, variance predictors and length regulator are used from that implementation. CHANGES.md lists each modified module. This is a FastSpeech 2 adaptation, not a paper-exact reproduction.')
    table([['Component','Implemented choice'],['Transformer','4 encoder / 6 decoder FFT layers; width 256; 2 heads; convolution width 1024; kernels 9 and 1'],['Variance module','Width 256; kernel 3; dropout 0.5; 256 linear bins; token-level pitch/energy'],['Training','Independent language runs; AdamW LR 0.0001; batch 8; gradient norm cap 1; seed 42; initial budget 150 epochs'],['Alignment','Two-layer bidirectional LSTM CTC; hidden 192 per direction; Adam LR 0.001; initial budget 40 epochs'],['Acoustics','22,050 Hz mono; FFT/window 1024; hop 256; 80 mel bands; 0-8000 Hz; pYIN F0 50-600 Hz']],[135,365])
    para('English uses a fixed ASCII-plus-punctuation inventory. Telugu adds the full U+0C00-U+0C7F block and retains combining marks, virama, and joiners. These are Unicode code-point tokens, not phonemes or complete orthographic syllables. Unsupported text raises an error and is logged during selection; it is never silently stripped. Both language frontends apply NFC and lowercase only. Original test text is retained separately from model input.')
    para('No language-specific numeric expansion, heteronym resolver or code-switch transliteration is asserted. Literal test anomalies are retained. Telugu character durations can partition dependent marks artificially; this limitation motivates future syllable/phoneme input with a vetted language aligner.')
    para('Implementation fixes include device-local length tensors, corrected energy control, padded-duration masking, duration scaling before rounding, and explicit failure for empty or excessive synthesized sequences. Separate configs, vocabularies, checkpoints and outputs prevent accidental cross-language training.')
    page('Dataset preparation and duration supervision')
    para('The exact assignment dataset identifiers are ai4bharat/Svarah and SPRINGLab/IndicTTS_Telugu. Svarah exposes an ASR test split; it is repurposed for this assignment only. Its Hugging Face schema lacks verified speaker IDs, so English uses a pooled voice. The Telugu card describes two speakers; its gender labels identify those two voices for this corpus. English pooled voice variability is an important confound.')
    para('Decoded recording hashes remove exact duplicate audio. Normalized identical transcripts form groups that remain wholly in one partition; both Telugu voices can therefore be retained without identical-text leakage between partitions. Groups are shuffled with seed 42, then accumulated to at least five training hours and one validation hour. Excess greater than 60 seconds fails. The method permits speaker overlap and evaluates held-out utterances, not unseen speakers. Rejected files, chosen IDs, waveform hashes and actual durations are recorded.')
    para('Waveforms are mixed to mono and resampled without silence trimming, preserving the timing axis. Log power mel features, pYIN F0 and RMS energy share the same STFT frame count. An independent CTC acoustic aligner learns transcript alignment on the training partition only. Monotonic Viterbi includes blank and repeated-label transitions. Every mel frame is assigned to a token; leading blanks attach to the first token and interior blanks to the preceding token. Duration sums must equal mel length.')
    para('The best aligner is chosen by greedy CER on a fixed training subset, not validation. CER above 0.35 aborts export. A low subset CER does not guarantee good alignments: inspect full per-utterance CER, log-probability, long-duration outliers and sample boundaries. This procedure is learned alignment supervision, not equal-duration pseudo-labels.')
    para('Token F0 averages voiced frames, with wholly unvoiced tokens assigned zero. Energy averages token RMS. Normalization means and standard deviations are computed solely on training token targets, including zero F0 tokens. Validation reuses those statistics. Predictions are inverse-transformed for plots; negative F0/energy are clipped for display and the raw values are retained in JSON.')
    rows=[['Language','Measured selected hours','Evidence']]
    for lang in ['English','Telugu']:
        m=maybe(ROOT/'data'/lang/'metadata.json')
        rows.append([lang,f"train={m['train_hours']:.4f}; val={m['validation_hours']:.4f}" if m else 'NOT MEASURED','data/'+lang+'/metadata.json' if m else 'Requires full data preparation'])
    table(rows,[85,195,220])
    page('Training and validation analysis')
    for lang in ['English','Telugu']:
        heading(lang)
        history=maybe(ROOT/'Results'/lang/'loss_history.json')
        if not history:
            para('NOT RUN. No claim of convergence, instability, overfitting or comparative performance is supported. After training, inspect component curves as well as total loss. A growing train-validation gap may indicate overfitting; abrupt joint spikes may indicate corrupted targets or optimization instability. These are interpretation criteria, not observations.')
        else:
            best=min(history,key=lambda r:r['validation']['total']);last=history[-1]
            para(f"Observed {len(history)} logged epochs. Best validation total {best['validation']['total']:.6f} at epoch {best['epoch']}. Latest train total {last['train']['total']:.6f}; latest validation total {last['validation']['total']:.6f}. These values are teacher-forced, size-weighted averages of masked batch means.")
            for name in ['training_loss.png','validation_loss.png']:
                path=ROOT/'Results'/lang/name
                if path.exists():story.append(Image(str(path),width=470,height=261))
    para('Checkpoint selection uses minimum validation total. last.pt supports resume and best.pt supports synthesis. Optimizer, completed epoch, RNG states, configuration, source split identity, vocabulary and statistics are saved or retained alongside the checkpoints. The data/architecture fingerprint rejects incompatible checkpoints.')
    page('Test sentence analysis and listening protocol')
    para('All five utterances per language are defined in the UTF-8 test_sentences.json files. The a/b pairs are separate outputs. Text from the rendered PDF is preserved, including the unusual internal spaces; glyph-layout artifacts from raw PDF extraction are not treated as linguistic edits. The JSON files are the authoritative transcript records in this project.')
    table([['ID','Question to evaluate after synthesis'],['E1','Does the long word remain intelligible, with plausible segment timing and no collapse?'],['E2a / E2b','Compare noun versus verb conduct: stress placement, vowel realization, duration, F0 and energy. Context may help but correct stress is not guaranteed.'],['E3','Does the literal split rhyth m create an unnatural break, omission or spelling-like output?'],['E4','Check time/date reading and the code-switched andaroo. No explicit normalization model is included.'],['T1','Evaluate the long technical loanword, dependent vowels, consonant clusters and overall articulation.'],['T2a / T2b','The word pattu has different meanings in context; semantic difference need not imply different phonemes. Compare contextual rhythm/prosody without inventing an expected phonemic contrast.'],['T3','Evaluate whether the deliberate split in the word for forest produces a pause or pronunciation error.'],['T4','Check numerals, ordinal suffix, time expression, phrasing and sentence ending.']],[85,415])
    notes=read_json(ROOT/'Report/listening_notes.json')
    para('Recorded listening/variance observations (NOT EVALUATED means no empirical claim):')
    if all(all(v=='NOT EVALUATED' for v in n.values()) for n in notes.values()):
        para('All ten sentences: pronunciation, quality, duration, pitch and energy are NOT EVALUATED. The listening_notes.json file provides an explicit field for every required observation.')
    else:
        for id,note in notes.items():
            para(id+': '+'; '.join(f'{k}: {v}' for k,v in note.items()))
    page('Variance evidence, comparison and reproducibility')
    for lang in ['English','Telugu']:
        summary=maybe(ROOT/'Results'/lang/'inference_summary.json')
        heading(lang+' variance evidence')
        if not summary:para('NOT GENERATED. Duration, pitch and energy plots require actual inference. No observed peaks, pauses, stress patterns or speech quality are inferred in their absence.')
        else:
            rows=[['ID','Audio seconds','Zero-duration tokens','Mean token F0 / RMS']]
            for r in summary:
                meanp=sum(r['pitch_hz'])/len(r['pitch_hz']);meane=sum(r['energy_rms'])/len(r['energy_rms'])
                rows.append([r['id'],f"{r['seconds']:.3f}",r['zero_duration_tokens'],f'{meanp:.2f} Hz / {meane:.5f}'])
            table(rows,[65,110,130,195])
    para('Plot x-axes are Unicode token indices, with exact text mappings in adjacent JSON. Token means cannot show within-token F0 movement. Zero F0 denotes an unvoiced target convention, not a measured phonation label for synthesized audio. Duration sums give the mel-frame count; centered-STFT reconstruction can differ by roughly one hop in waveform duration. Compare inverse-transformed units within matched voices, and account for voice F0 ranges before comparing languages.')
    para('No English-versus-Telugu winner can be claimed without the runs. Expected confounds include multi-speaker pooled English versus two conditioned Telugu voices, noisier ASR-source recordings versus studio TTS recordings, alignment quality, character segmentation, text normalization and the reconstruction method. Griffin-Lim artifacts must be separated from model pronunciation failures through mel/variance inspection and listening. Improvements include verified speaker labels, vetted phoneme alignments, syllable-aware Telugu input, contextual numeric normalization and a matching trained neural vocoder.')
    heading('Reproducible commands')
    for lang in ['English','Telugu']:
        para(f'python {lang}/preprocess.py --manifest data/raw/{lang}/manifest.json')
        para(f'python -m fs2.alignment --config {lang}/config.yaml --device auto')
        para(f'python {lang}/train.py --device auto; then python {lang}/inference.py --device auto')
    para('Run python scripts/smoke_test.py first. See README.md for virtual environments, CPU/CUDA installation, dataset export, licenses, resume, output paths and GitHub submission. After listening, edit Report/listening_notes.json and run python scripts/build_report.py followed by python scripts/check_submission.py. Software ranges are in requirements.txt; the authoring test environment is recorded in TESTING.md.')
    heading('References')
    for x in ['Ren et al. FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. https://arxiv.org/abs/2006.04558','ming024/FastSpeech2. https://github.com/ming024/FastSpeech2 (supplied source archive)','AI4Bharat Svarah: https://huggingface.co/datasets/ai4bharat/Svarah and https://github.com/AI4Bharat/Svarah','SPRINGLab IndicTTS Telugu: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu','Original IndicTTS license: https://www.iitm.ac.in/donlab/indictts/downloads/license.pdf']:
        para(x)
    output=ROOT/'Report/Technical_Report.pdf'
    def footer(canvas,doc):
        canvas.setFont('ReportSans',8);canvas.setFillColor(colors.HexColor('#617583'));canvas.drawString(48,28,'FastSpeech 2 | Implementation and evidence status');canvas.drawRightString(548,28,str(doc.page))
    SimpleDocTemplate(str(output),pagesize=(595.28,841.89),rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=48,title='FastSpeech 2 — English and Telugu Technical Report',author='Implementation package').build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)
if __name__=='__main__':build()
