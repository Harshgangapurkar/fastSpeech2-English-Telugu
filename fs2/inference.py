"""Free-running synthesis and variance curves from actual checkpoint predictions."""
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from .common import ROOT, config, device, seed_all, encode, read_json, write_json, normalize, fingerprint
from .train import build_model

def invert_mel(mel,a):
    # Same power mel filter and natural log convention as preprocessing.
    power=np.exp(np.clip(mel.T,-11.512925,12))
    linear=librosa.feature.inverse.mel_to_stft(power,sr=a['sample_rate'],n_fft=a['n_fft'],power=2.0,fmin=a['fmin'],fmax=a['fmax'])
    y=librosa.griffinlim(linear,n_iter=64,hop_length=a['hop_length'],win_length=a['win_length'],n_fft=a['n_fft'],random_state=42)
    peak=np.max(np.abs(y))
    return y/max(peak,1.0)*.95

def variance_plots(out,id,duration,pitch,energy,sr,hop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for key,values,label in [('duration',duration*hop/sr,'Duration (seconds)'),('pitch',pitch,'Token mean F0 (Hz; 0 = unvoiced)'),('energy',energy,'Token RMS amplitude')]:
        folder=Path(out)/'variance_adaptor_plots'/key;folder.mkdir(parents=True,exist_ok=True)
        fig,ax=plt.subplots(figsize=(11,3.5));ax.plot(np.arange(len(values)),values,marker='.',linewidth=1)
        ax.set(title=f'{id}: predicted {key}',xlabel='Unicode token index (see adjacent JSON for text mapping)',ylabel=label);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(folder/f'{id}.png',dpi=150);plt.close(fig)

def synthesize(c,checkpoint,dev,sentences=None,speaker=None):
    seed_all(c['seed']);root=Path(c['data_dir']);state=torch.load(checkpoint,map_location=dev,weights_only=False)
    if state['fingerprint']!=fingerprint(c,read_json(root/'metadata.json')): raise ValueError('Checkpoint/data/config mismatch')
    model=build_model(c).to(dev);model.load_state_dict(state['model']);model.eval()
    speakers=read_json(root/'speakers.json');vocab=read_json(root/'vocab.json');stats=state['stats']
    if speaker is None:
        rows=read_json(root/'train.json');seconds={s:sum(r['seconds'] for r in rows if r['speaker']==s) for s in speakers};speaker=max(seconds,key=seconds.get)
    if speaker not in speakers: raise ValueError(f'Unknown speaker {speaker}; choices: {list(speakers)}')
    sentences=sentences or read_json(ROOT/c['language']/'test_sentences.json');out=Path(c['results_dir']);audio=out/'test_generated_audios';audio.mkdir(parents=True,exist_ok=True)
    summaries=[]
    with torch.no_grad():
        for row in sentences:
            id=row['id']
            if not id.replace('_','').replace('-','').isalnum(): raise ValueError('Unsafe output id')
            tokens=torch.from_numpy(encode(row['text'],vocab))[None].to(dev);lens=torch.tensor([tokens.shape[1]],device=dev)
            pred=model(torch.tensor([speakers[speaker]],device=dev),tokens,lens,tokens.shape[1]);mel=pred[1][0,:int(pred[9][0])].cpu().numpy()
            if not np.isfinite(mel).all(): raise ValueError('Nonfinite generated mel')
            d=pred[5][0].cpu().numpy().astype(int);p=pred[2][0].cpu().numpy()*stats['pitch'][3]+stats['pitch'][2];e=pred[3][0].cpu().numpy()*stats['energy'][3]+stats['energy'][2]
            rawp=p.copy();rawe=e.copy();p=np.maximum(p,0);e=np.maximum(e,0)
            y=invert_mel(mel,c['audio']);sf.write(audio/f'{id}.wav',y,c['audio']['sample_rate'],subtype='PCM_16')
            np.save(audio/f'{id}.mel.npy',mel);variance_plots(out,id,d,p,e,c['audio']['sample_rate'],c['audio']['hop_length'])
            data={'id':id,'text':row['text'],'model_text':normalize(row['text']),'speaker':speaker,'tokens':list(normalize(row['text'])),'duration_frames':d.tolist(),'pitch_hz':p.tolist(),'energy_rms':e.tolist(),'raw_pitch_hz':rawp.tolist(),'raw_energy_rms':rawe.tolist(),'vocoder':'Griffin-Lim 64 iterations; not a neural vocoder','seconds':len(y)/c['audio']['sample_rate'],'checkpoint_epoch':state['epoch'],'zero_duration_tokens':int((d==0).sum())}
            write_json(audio/f'{id}.json',data);summaries.append(data);print(f'generated {id}: {data["seconds"]:.2f}s',flush=True)
    write_json(out/'inference_summary.json',summaries)

def main(default):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=default);p.add_argument('--checkpoint');p.add_argument('--device',default='auto');p.add_argument('--speaker');p.add_argument('--sentences',help='Optional JSON [{id,text}, ...]')
    a=p.parse_args();c=config(a.config);synthesize(c,a.checkpoint or Path(c['checkpoint_dir'])/'best.pt',device(a.device),read_json(a.sentences) if a.sentences else None,a.speaker)
