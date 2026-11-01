#!/usr/bin/env python3
"""Neural semantic-paraphrase robustness test for a platform evaluation CSV."""
from __future__ import annotations
import argparse,csv,json,os,re,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel,AutoModelForSeq2SeqLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream'))
from watermark_processor import WatermarkDetector
PARAPHRASE_MODEL='Vamsi/T5_Paraphrase_Paws'
SIMILARITY_MODEL='sentence-transformers/all-MiniLM-L6-v2'
PROMPTS=[
'Artificial intelligence is changing education because',
'Renewable energy technologies can help cities by',
'A reliable software development process should',
'Climate change affects agriculture through',
'The benefits of scientific research include',
'Modern communication networks are important because',
'Students can improve their learning outcomes when',
'A secure computer system should always',
'Machine learning models are useful for',
'The future of transportation may depend on']
def mean_pool(hidden,mask):
    mask=mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
def jaccard(a,b):
    x=set(re.findall(r'[a-z]+',a.lower()));y=set(re.findall(r'[a-z]+',b.lower()))
    return len(x&y)/len(x|y) if x|y else 0.0
def detect(detector,tokenizer,text,prompt):
    ids=tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    pids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    if pids.numel(): ids=torch.cat((pids[-1:],ids))
    try:return detector.detect(tokenized_text=ids)
    except (ValueError,ZeroDivisionError,AssertionError,IndexError):return {'prediction':False,'z_score':None,'p_value':1.0,'num_tokens_scored':0}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output-dir',default=str(ROOT/'results'));ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--max-samples',type=int,default=0);ap.add_argument('--model',default='distilgpt2');ap.add_argument('--gamma',type=float,default=.5);ap.add_argument('--delta',type=float,default=2.0);ap.add_argument('--z-threshold',type=float,default=4.0);args=ap.parse_args()
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    df=pd.read_csv(args.input);df=df[df.is_watermarked==1].copy()
    if args.max_samples:df=df.head(args.max_samples)
    if df.empty:raise SystemExit('No watermarked samples found in input CSV')
    print('Loading T5 paraphraser...',flush=True);ptok=AutoTokenizer.from_pretrained(PARAPHRASE_MODEL);pmodel=AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to('cpu').eval()
    print('Loading MiniLM similarity encoder...',flush=True);stok=AutoTokenizer.from_pretrained(SIMILARITY_MODEL);smodel=AutoModel.from_pretrained(SIMILARITY_MODEL).to('cpu').eval()
    source_model=args.model
    # The evaluator currently records a single selected model; keep it explicit in this report.
    from transformers import AutoTokenizer as AT
    wtok=AT.from_pretrained(source_model);vocab=list(range(len(wtok)))
    key=int(os.environ['WATERMARK_RESEARCH_KEY']);gamma=args.gamma
    detector=WatermarkDetector(vocab=vocab,gamma=gamma,delta=args.delta,hash_key=key,device=torch.device('cpu'),tokenizer=wtok,z_threshold=args.z_threshold,normalizers=[],ignore_repeated_bigrams=True)
    originals=df.text.astype(str).tolist();prompts=[PROMPTS[int(i)%len(PROMPTS)] for i in df.prompt_id]
    paras=[]
    for start in range(0,len(originals),args.batch_size):
        batch=originals[start:start+args.batch_size]
        enc=ptok(['paraphrase: '+t+' </s>' for t in batch],return_tensors='pt',padding=True,truncation=True,max_length=256)
        with torch.inference_mode():out=pmodel.generate(input_ids=enc.input_ids,attention_mask=enc.attention_mask,max_new_tokens=160,do_sample=True,top_k=120,top_p=.95,temperature=.9)
        paras.extend(ptok.batch_decode(out,skip_special_tokens=True,clean_up_tokenization_spaces=True))
        print(f'Paraphrased {min(start+len(batch),len(originals))}/{len(originals)}',flush=True)
    records=[]
    for i,(orig,para,prompt) in enumerate(zip(originals,paras,prompts)):
        od=detect(detector,wtok,orig,prompt);pd_=detect(detector,wtok,para,prompt)
        enc=stok([orig,para],return_tensors='pt',padding=True,truncation=True,max_length=256)
        with torch.inference_mode():
            emb=mean_pool(smodel(**enc).last_hidden_state,enc.attention_mask);emb=torch.nn.functional.normalize(emb,p=2,dim=1)
        records.append({'prompt_id':int(df.iloc[i].prompt_id),'seed':int(df.iloc[i].seed),'original_text':orig,'paraphrased_text':para,'original_detected':int(bool(od.get('prediction'))),'paraphrase_detected':int(bool(pd_.get('prediction'))),'original_z_score':od.get('z_score'),'paraphrase_z_score':pd_.get('z_score'),'paraphrase_tokens_scored':pd_.get('num_tokens_scored'),'semantic_cosine_similarity':float((emb[0]*emb[1]).sum()),'lexical_jaccard':jaccard(orig,para)})
        print(f'Evaluated {i+1}/{len(originals)}',flush=True)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    csvpath=out/f'platform_semantic_paraphrase_{rid}.csv';pd.DataFrame(records).to_csv(csvpath,index=False)
    valid=[r for r in records if r['paraphrase_tokens_scored']>0]
    report={'run_id':rid,'created_utc':datetime.now(timezone.utc).isoformat(),'source_csv':str(Path(args.input).resolve()),'paraphrase_model':PARAPHRASE_MODEL,'similarity_model':SIMILARITY_MODEL,'model_id':source_model,'watermark_parameters':{'gamma':gamma,'delta':args.delta,'z_threshold':args.z_threshold},'n':len(records),'original_detection_rate':float(np.mean([r['original_detected'] for r in records])),'paraphrase_detection_rate':float(np.mean([r['paraphrase_detected'] for r in valid])) if valid else None,'scorable_paraphrases':len(valid),'mean_semantic_cosine_similarity':float(np.mean([r['semantic_cosine_similarity'] for r in records])),'median_semantic_cosine_similarity':float(np.median([r['semantic_cosine_similarity'] for r in records])),'mean_lexical_jaccard':float(np.mean([r['lexical_jaccard'] for r in records])),'artifacts':{'csv':str(csvpath)},'caveats':['Automatic cosine similarity is a proxy, not human semantic evaluation.','T5 paraphraser may change meaning or produce malformed outputs; inspect samples.','This is a small exploratory sample; increase prompt and seed counts.']}
    jp=out/f'platform_semantic_paraphrase_report_{rid}.json';report['artifacts']['report_json']=str(jp);jp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
