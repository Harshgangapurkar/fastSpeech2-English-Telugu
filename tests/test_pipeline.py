"""Meaningful contract checks; synthetic fixtures are never submission results."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fs2.common import config, ROOT, vocabulary, encode, write_json, fingerprint
from fs2.preprocess import select, features
from fs2.alignment import viterbi, Aligner
from fs2.train import build_model, collate, forward, losses, fit
from fs2.inference import synthesize

class Contracts(unittest.TestCase):
    def test_ctc_repeats_and_coverage(self):
        logits=np.full((7,3),-20.0)
        for t,label in enumerate([0,1,1,0,1,2,0]): logits[t,label]=0
        d,score=viterbi(logits,[1,1,2]);self.assertEqual(d.tolist(),[4,1,2]);self.assertEqual(score,0)
        with self.assertRaises(ValueError):viterbi(np.zeros((2,3)),[1,1])
    def test_telugu_preserved(self):
        s='పులి అడ విలో';v=vocabulary('Telugu');ids=encode(s,v);rev={i:t for t,i in v.items()};self.assertEqual(''.join(rev[i] for i in ids),s)
        with self.assertRaises(ValueError):encode('🙂',v)
    def test_split_duplicate_protection(self):
        rows=[{'id':str(i),'audio_hash':str(i),'text':f'word {i}','seconds':1} for i in range(20)]
        rows.append(dict(rows[0],id='duplicate'))
        a,b=select(rows,5,3,42);a2,b2=select(rows,5,3,42);self.assertEqual(a,a2)
        self.assertFalse({r['audio_hash'] for r in a}&{r['audio_hash'] for r in b});self.assertFalse({r['text'] for r in a}&{r['text'] for r in b})
        with self.assertRaises(ValueError):select(rows,30,5,42)
    def test_aligner_backward(self):
        torch.manual_seed(42);m=Aligner(5,8);x=torch.randn(2,16,80);lens=torch.tensor([16,12]);out=m(x,lens)
        loss=torch.nn.CTCLoss()(out.transpose(0,1),torch.tensor([1,2,2,3]),lens,torch.tensor([2,2]));loss.backward();self.assertTrue(torch.isfinite(loss))
    def test_real_audio_features(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'tone.wav';sr=22050;t=np.arange(sr)/sr;sf.write(p,.2*np.sin(2*np.pi*180*t),sr)
            c=config(ROOT/'English/config.yaml');mel,pitch,energy=features(p,c['audio']);self.assertEqual(mel.shape[1],80);self.assertEqual(len(mel),len(pitch));self.assertEqual(len(pitch),len(energy));self.assertTrue(np.isfinite(mel).all());self.assertAlmostEqual(float(np.median(pitch[pitch>0])),180,delta=8)
    def test_alignment_export_and_training_only_statistics(self):
        from fs2.alignment import fit_and_extract
        from fs2.common import read_json
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            c=config(ROOT/'English/config.yaml');root=Path(folder);c['data_dir']=str(root)
            c['alignment'].update(hidden=8,batch_size=2,max_train_cer=2.0)
            # The relaxed gate is for random synthetic fixtures ONLY, never real configs.
            (root/'features').mkdir();v=vocabulary('English');write_json(root/'vocab.json',v)
            rows=[{'id':'a'},{'id':'b'}];write_json(root/'train.json',rows[:1]);write_json(root/'validation.json',rows[1:]);write_json(root/'metadata.json',{'synthetic_test_only':True})
            for i,row in enumerate(rows):
                np.savez_compressed(root/'features'/f"{row['id']}.npz",tokens=encode('ab',v),mel=np.zeros((12,80),dtype='float32'),pitch=np.full(12,100 if i==0 else 500,dtype='float32'),energy=np.full(12,.1 if i==0 else .9,dtype='float32'),speaker=np.array(0))
            fit_and_extract(c,torch.device('cpu'),epochs=1)
            stats=read_json(root/'stats.json');self.assertEqual(stats['pitch'][2],100)
            with np.load(root/'aligned/b.npz') as z:self.assertEqual(int(z['duration'].sum()),12)

    def test_training_resume_synthesis_both_languages(self):
        torch.set_num_threads(2)
        for lang,text in [('English','ab'),('Telugu','పట')]:
            with tempfile.TemporaryDirectory() as folder:
                c=config(ROOT/lang/'config.yaml');root=Path(folder);data=root/'data';data.mkdir();(data/'aligned').mkdir()
                c.update(data_dir=str(data),checkpoint_dir=str(root/'checkpoints'),results_dir=str(root/'results'))
                mc=c['model'];mc['max_seq_len']=128;mc['transformer'].update(encoder_layer=1,decoder_layer=1,encoder_hidden=16,decoder_hidden=16,conv_filter_size=32);mc['variance_predictor']['filter_size']=16
                c['training'].update(batch_size=2,num_workers=0);v=vocabulary(lang);write_json(data/'vocab.json',v);write_json(data/'speakers.json',{'fixture':0});write_json(data/'stats.json',{'pitch':[-2,2,150,30],'energy':[-2,2,.1,.05]})
                tokens=encode(text,v);n=len(tokens);T=n*3
                rows=[]
                for i in range(3):
                    id=f'fixture_{i}';rows.append({'id':id,'text':text,'speaker':'fixture','seconds':1})
                    np.savez_compressed(data/'aligned'/f'{id}.npz',tokens=tokens,mel=np.random.default_rng(i).normal(-3,1,(T,80)).astype('float32'),pitch=np.zeros(n,dtype='float32'),energy=np.zeros(n,dtype='float32'),duration=np.full(n,3,dtype='int64'),speaker=np.array(0,dtype='int64'))
                write_json(data/'train.json',rows[:2]);write_json(data/'validation.json',rows[2:]);meta={'synthetic_test_only':True};write_json(data/'metadata.json',meta);write_json(data/'alignment_metadata.json',{'fingerprint':fingerprint(c,meta)})
                fit(c,torch.device('cpu'),epochs=1);fit(c,torch.device('cpu'),epochs=2,resume=root/'checkpoints/last.pt')
                state=torch.load(root/'checkpoints/best.pt',weights_only=False);self.assertTrue(np.isfinite(state['history'][-1]['train']['total']))
                # Force nonzero duration for deterministic export mechanics only, not speech quality.
                for key in list(state['model']):
                    if 'duration_predictor' in key: state['model'][key].zero_()
                state['model']['variance_adaptor.duration_predictor.linear_layer.bias'].fill_(np.log(4))
                ck=root/'fixture.pt';torch.save(state,ck)
                synthesize(c,ck,torch.device('cpu'),[{'id':'fixture','text':text}])
                self.assertTrue((root/'results/test_generated_audios/fixture.wav').is_file());self.assertTrue((root/'results/variance_adaptor_plots/duration/fixture.png').is_file())
                # Padded duration predictions must remain zero even with positive bias.
                model=build_model(c);model.load_state_dict(state['model']);model.eval()
                with torch.no_grad():
                    pred=model(torch.tensor([0,0]),torch.tensor([[tokens[0],tokens[1]],[tokens[0],0]]),torch.tensor([2,1]),2)
                self.assertEqual(pred[5][1,1].item(),0)
if __name__=='__main__':unittest.main(verbosity=2)
