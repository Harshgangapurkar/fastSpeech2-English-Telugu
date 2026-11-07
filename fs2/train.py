"""Separate FastSpeech 2 training, masked losses, resume and real loss plots."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .common import config, seed_all, device, read_json, write_json, fingerprint
from .model import FastSpeech2

class SpeechDataset(Dataset):
    def __init__(self,root,split): self.root=Path(root);self.rows=read_json(self.root/f'{split}.json')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        with np.load(self.root/'aligned'/f"{self.rows[index]['id']}.npz") as z:
            return {k:torch.from_numpy(z[k]) for k in ['mel','tokens','pitch','energy','duration','speaker']}

def collate(items):
    batch={k:pad_sequence([i[k] for i in items],batch_first=True) for k in ['mel','tokens','pitch','energy','duration']}
    batch['speaker']=torch.stack([i['speaker'] for i in items]).long()
    batch['src_lens']=torch.tensor([len(i['tokens']) for i in items]);batch['mel_lens']=torch.tensor([len(i['mel']) for i in items])
    return batch

def build_model(c):
    mc=copy.deepcopy(c['model']);mc['vocab_size']=len(read_json(Path(c['data_dir'])/'vocab.json'))+1
    pc={'path':{'preprocessed_path':c['data_dir']},'preprocessing':{'mel':{'n_mel_channels':80},'pitch':{'feature':'phoneme_level'},'energy':{'feature':'phoneme_level'}}}
    # Upstream calls this phoneme_level; our units are Unicode code points, explicitly documented.
    return FastSpeech2(pc,mc)

def forward(model,b):
    return model(b['speaker'],b['tokens'],b['src_lens'],b['tokens'].shape[1],mels=b['mel'],mel_lens=b['mel_lens'],max_mel_len=b['mel'].shape[1],p_targets=b['pitch'],e_targets=b['energy'],d_targets=b['duration'])

def losses(pred,b):
    valid_src=~pred[6];valid_mel=(~pred[7]).unsqueeze(-1).expand_as(b['mel'])
    result={'mel':(pred[0]-b['mel']).abs()[valid_mel].mean(),'postnet':(pred[1]-b['mel']).abs()[valid_mel].mean(),'pitch':((pred[2]-b['pitch'])**2)[valid_src].mean(),'energy':((pred[3]-b['energy'])**2)[valid_src].mean(),'duration':((pred[4]-torch.log1p(b['duration'].float()))**2)[valid_src].mean()}
    result['total']=sum(result.values());return result

def plot_history(history,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for split,file in [('train','training_loss.png'),('validation','validation_loss.png')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for key in ['total','mel','postnet','duration','pitch','energy']: ax.plot([r['epoch'] for r in history],[r[split][key] for r in history],label=key)
        ax.set(xlabel='Epoch',ylabel='Masked mean loss',title=f'{split.title()} losses (teacher-forced)');ax.legend(ncol=3);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(destination/file,dpi=150);plt.close(fig)

def fit(c,dev,epochs=None,resume=None):
    seed_all(c['seed']);root=Path(c['data_dir']);meta=read_json(root/'metadata.json');fp=fingerprint(c,meta)
    am=read_json(root/'alignment_metadata.json')
    if am['fingerprint']!=fp: raise ValueError('Configuration differs from aligned data; rerun alignment with the intended configuration')
    model=build_model(c).to(dev);tc=c['training'];opt=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc['weight_decay'])
    history=[];start=1;best=float('inf');steps=0
    if resume:
        state=torch.load(resume,map_location=dev,weights_only=False)
        if state['fingerprint']!=fp: raise ValueError('Checkpoint/data/config mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start=state['epoch']+1;best=state['best'];history=state['history'];steps=state['steps']
        torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng'):torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    loaders={s:DataLoader(SpeechDataset(root,s),batch_size=tc['batch_size'],shuffle=s=='train',collate_fn=collate,num_workers=tc['num_workers']) for s in ['train','validation']}
    ck=Path(c['checkpoint_dir']);ck.mkdir(parents=True,exist_ok=True);results=Path(c['results_dir']);results.mkdir(parents=True,exist_ok=True)
    for epoch in range(start,(epochs or tc['epochs'])+1):
        record={'epoch':epoch}
        for split,loader in loaders.items():
            model.train(split=='train');sums={};n=0
            for b in loader:
                b={k:v.to(dev) for k,v in b.items()}
                with torch.set_grad_enabled(split=='train'):
                    loss=losses(forward(model,b),b)
                    if not torch.isfinite(loss['total']): raise ValueError('Nonfinite loss; inspect features/alignment')
                    if split=='train':
                        opt.zero_grad();loss['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),tc['grad_clip']);opt.step();steps+=1
                size=len(b['speaker']);n+=size
                for key,value in loss.items(): sums[key]=sums.get(key,0)+float(value.detach())*size
            record[split]={k:v/n for k,v in sums.items()}
        improved=record['validation']['total']<best;best=min(best,record['validation']['total']);history.append(record)
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'steps':steps,'best':best,'history':history,'config':c,'fingerprint':fp,'metadata':meta,'stats':read_json(root/'stats.json'),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        tmp=ck/'last.tmp';torch.save(state,tmp);tmp.replace(ck/'last.pt')
        if improved: torch.save(state,ck/'best.pt')
        write_json(results/'loss_history.json',history);plot_history(history,results)
        (ck/'NOT_TRAINED.txt').unlink(missing_ok=True)
        (results/'STATUS.md').write_text(f'Trained through epoch {epoch}. Best teacher-forced validation total: {best:.6f}. Inference/listening evaluation is a separate step.\n')
        print(f"epoch {epoch}: train={record['train']['total']:.5f} validation={record['validation']['total']:.5f}",flush=True)
    return model

def main(default):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=default);p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int);p.add_argument('--resume')
    a=p.parse_args();fit(config(a.config),device(a.device),a.epochs,a.resume)
