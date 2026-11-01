#!/usr/bin/env python3
"""Run a genuine T5 semantic-paraphrase attack on saved watermarked samples.

Measures watermark detection before/after paraphrasing, semantic similarity
with MiniLM sentence embeddings, and lexical Jaccard overlap.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'upstream'))
from extended_watermark_processor import WatermarkDetector

PARAPHRASE_MODEL = 'Vamsi/T5_Paraphrase_Paws'
SIMILARITY_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
SOURCE_MODEL = 'distilgpt2'

def detect_safe(detector, text):
    try:
        return detector.detect(text)
    except ValueError as exc:
        if 'Must have at least' not in str(exc): raise
        return {'num_tokens_scored': 0, 'green_fraction': float('nan'),
                'z_score': float('nan'), 'p_value': 1.0, 'prediction': False}

def mean_pool(last_hidden, mask):
    mask = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def lexical_jaccard(a, b):
    ta = set(re.findall(r"[a-z]+", a.lower())); tb = set(re.findall(r"[a-z]+", b.lower()))
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, help='samples_*.csv from a completed baseline run')
    ap.add_argument('--output-dir', default=str(ROOT / 'results'))
    ap.add_argument('--max-samples', type=int, default=0, help='0 means all watermarked samples')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    df = pd.read_csv(args.input)
    df = df[df['is_watermarked'] == 1].copy()
    if args.max_samples: df = df.head(args.max_samples)
    df = df[df['text'].fillna('').str.split().str.len() >= 12].copy()
    if df.empty: raise SystemExit('No watermarked source texts with at least 12 words.')
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'Loading semantic paraphraser {PARAPHRASE_MODEL} on CPU', flush=True)
    ptok = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
    pmodel = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to('cpu').eval()
    print(f'Loading semantic similarity encoder {SIMILARITY_MODEL} on CPU', flush=True)
    stok = AutoTokenizer.from_pretrained(SIMILARITY_MODEL)
    smodel = AutoModel.from_pretrained(SIMILARITY_MODEL).to('cpu').eval()
    print(f'Loading watermark detector tokenizer {SOURCE_MODEL}', flush=True)
    wtok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    vocab = list(wtok.get_vocab().values())
    key = int(os.environ.get('WATERMARK_RESEARCH_KEY', '15485863'))
    scheme = f'ff-anchored_minhash_prf-4-True-{key}'
    detector = WatermarkDetector(vocab=vocab, gamma=0.25, seeding_scheme=scheme,
        device=torch.device('cpu'), tokenizer=wtok, z_threshold=4.0,
        normalizers=[], ignore_repeated_ngrams=True)
    source_texts = df['text'].astype(str).tolist()
    paraphrases = []
    for start in range(0, len(source_texts), args.batch_size):
        batch = source_texts[start:start + args.batch_size]
        prompts = ['paraphrase: ' + x + ' </s>' for x in batch]
        enc = ptok(prompts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            gen = pmodel.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                max_new_tokens=160, do_sample=True, top_k=120, top_p=0.95,
                temperature=0.9, num_return_sequences=1)
        paraphrases.extend(ptok.batch_decode(gen, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True))
        print(f'Paraphrased {min(start+len(batch),len(source_texts))}/{len(source_texts)}', flush=True)
    records=[]
    for i,( (_, row), original, para) in enumerate(zip(df.iterrows(), source_texts, paraphrases)):
        od=detect_safe(detector, original); pd_=detect_safe(detector, para)
        # MiniLM cosine similarity, computed in a pair batch.
        enc=stok([original,para], return_tensors='pt', padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            emb=mean_pool(smodel(**enc).last_hidden_state, enc['attention_mask'])
            emb=torch.nn.functional.normalize(emb, p=2, dim=1)
            similarity=float((emb[0]*emb[1]).sum().item())
        records.append({
            'source_run_id': row.get('run_id'), 'prompt_id': row.get('prompt_id'),
            'seed': row.get('seed'), 'max_new_tokens': row.get('max_new_tokens'),
            'original_text': original, 'paraphrased_text': para,
            'original_word_count': len(original.split()), 'paraphrase_word_count': len(para.split()),
            'original_z_score': od.get('z_score'), 'paraphrase_z_score': pd_.get('z_score'),
            'original_detected': int(bool(od.get('prediction'))),
            'paraphrase_detected': int(bool(pd_.get('prediction'))),
            'paraphrase_num_tokens_scored': pd_.get('num_tokens_scored'),
            'semantic_cosine_similarity': similarity,
            'lexical_jaccard': lexical_jaccard(original,para),
        })
        if (i+1)%10==0 or i+1==len(source_texts): print(f'Evaluated {i+1}/{len(source_texts)}',flush=True)
    outcsv=outdir/f'semantic_paraphrase_{run_id}.csv'
    pd.DataFrame(records).to_csv(outcsv,index=False)
    res=pd.DataFrame(records)
    valid=res[res.paraphrase_num_tokens_scored.fillna(0)>0]
    report={
      'run_id':run_id,'created_utc':datetime.now(timezone.utc).isoformat(),
      'source_samples_csv':str(Path(args.input).resolve()),
      'source_model':SOURCE_MODEL,'paraphrase_model':PARAPHRASE_MODEL,
      'semantic_similarity_model':SIMILARITY_MODEL,'device':'cpu','n_source_samples':len(res),
      'watermark_threshold':4.0,
      'original_detection_rate':float(res.original_detected.mean()),
      'paraphrase_detection_rate_all':float(res.paraphrase_detected.mean()),
      'paraphrase_detection_rate_scorable':float(valid.paraphrase_detected.mean()) if len(valid) else None,
      'paraphrase_insufficient_text_n':int(len(res)-len(valid)),
      'mean_semantic_cosine_similarity':float(res.semantic_cosine_similarity.mean()),
      'median_semantic_cosine_similarity':float(res.semantic_cosine_similarity.median()),
      'mean_lexical_jaccard':float(res.lexical_jaccard.mean()),
      'mean_original_words':float(res.original_word_count.mean()),
      'mean_paraphrase_words':float(res.paraphrase_word_count.mean()),
      'artifacts':{'csv':str(outcsv)},
      'caveats':['MiniLM cosine similarity is an automatic semantic-preservation proxy, not human evaluation.',
        'PAWS-trained T5 paraphraser may fail or change meaning on some inputs; inspect examples.',
        'Paraphrase detection rate is measured at the fixed z-score threshold 4.0.',
        'This attack evaluates already-generated watermarked text; it does not regenerate control texts.']
    }
    outjson=outdir/f'semantic_paraphrase_report_{run_id}.json'
    outjson.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('\n=== SEMANTIC PARAPHRASE RESULTS ===',flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print(f'Saved CSV: {outcsv}\nSaved report: {outjson}',flush=True)

if __name__=='__main__': main()
