"""Paired control/watermarked evaluation using this application's exact watermark scheme."""
from __future__ import annotations
import csv, json, math, random, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
from . import model_manager
UPSTREAM = Path(__file__).resolve().parents[2] / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import LogitsProcessor
from .watermark_engine import _config
from src.attacks import apply_attack

PROMPTS = [
    "Artificial intelligence is changing education because",
    "Renewable energy technologies can help cities by",
    "A reliable software development process should",
    "Climate change affects agriculture through",
    "The benefits of scientific research include",
    "Modern communication networks are important because",
    "Students can improve their learning outcomes when",
    "A secure computer system should always",
    "Machine learning models are useful for",
    "The future of transportation may depend on",
]
ATTACKS = ["none", "word_deletion", "word_substitution", "lexical_paraphrase_proxy"]

class _SuppressEos(LogitsProcessor):
    """Keep sample lengths comparable by preventing premature EOS termination."""
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id
    def __call__(self, input_ids, scores):
        if self.eos_token_id is not None:
            scores[:, self.eos_token_id] = -float("inf")
        return scores


def _safe_detect(detector, text: str, token_ids=None) -> dict[str, Any]:
    try:
        if token_ids is not None:
            return detector.detect(tokenized_text=token_ids)
        return detector.detect(text)
    except (ValueError, ZeroDivisionError, AssertionError, IndexError):
        return {"num_tokens_scored": 0, "num_green_tokens": 0,
                "green_fraction": None, "z_score": None, "p_value": 1.0,
                "prediction": False, "insufficient_text": True}


def _rate_ci(values: list[int], seed: int = 2026, draws: int = 2000) -> list[float] | None:
    """Wilson score interval; remains informative for all-zero/all-one small samples."""
    n=len(values)
    if not n: return None
    z=1.959963984540054
    p=sum(values)/n
    denom=1+(z*z/n)
    centre=(p+(z*z/(2*n)))/denom
    margin=(z*math.sqrt((p*(1-p)/n)+(z*z/(4*n*n))))/denom
    lower=max(0.0,centre-margin)
    upper=min(1.0,centre+margin)
    if lower < 1e-12: lower=0.0
    if upper > 1.0-1e-12: upper=1.0
    return [float(lower),float(upper)]


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp=sum(r['prediction']==1 and r['label']==1 for r in rows)
    tn=sum(r['prediction']==0 and r['label']==0 for r in rows)
    fp=sum(r['prediction']==1 and r['label']==0 for r in rows)
    fn=sum(r['prediction']==0 and r['label']==1 for r in rows)
    div=lambda a,b: float(a/b) if b else None
    precision=div(tp,tp+fp); recall=div(tp,tp+fn); specificity=div(tn,tn+fp)
    f1=div(2*precision*recall,precision+recall) if precision is not None and recall is not None and precision+recall else None
    return {"n":len(rows),"true_positive":tp,"true_negative":tn,"false_positive":fp,"false_negative":fn,
            "accuracy":div(tp+tn,len(rows)),"precision":precision,"recall":recall,
            "specificity":specificity,"f1":f1,
            "false_positive_rate":div(fp,fp+tn),"detection_rate":recall}


def run_evaluation(model_id: str, prompt_count: int = 4, seeds: list[int] | None = None,
                   max_new_tokens: int = 96, gamma: float = .5, delta: float = 2.,
                   z_threshold: float = 4., attacks: list[str] | None = None) -> dict[str, Any]:
    if not 1 <= prompt_count <= len(PROMPTS): raise ValueError(f"prompt_count must be 1-{len(PROMPTS)}")
    seeds=seeds or [11,22]
    if not 1 <= len(seeds) <= 10: raise ValueError("Use between 1 and 10 seeds")
    if not 16 <= max_new_tokens <= 256: raise ValueError("max_new_tokens must be between 16 and 256 for meaningful evaluation")
    attacks=attacks or ATTACKS
    if not attacks or set(attacks)-set(ATTACKS): raise ValueError("Unsupported attack requested")
    state=model_manager._loaded
    if not state or state['model_id'] != model_id: raise RuntimeError("Load the selected Hugging Face model first")
    params,key=_config(gamma,delta,z_threshold)
    tokenizer,model=state['tokenizer'],state['model']
    device=next(model.parameters()).device
    vocab=list(range(int(model.config.vocab_size)))
    detector=WatermarkDetector(vocab=vocab,gamma=gamma,delta=delta,hash_key=key,
        device=device,tokenizer=tokenizer,z_threshold=z_threshold,normalizers=[],ignore_repeated_bigrams=True)
    outdir=Path(__file__).resolve().parents[2]/'results'; outdir.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    sample_rows=[]; attack_rows=[]
    for pi,prompt in enumerate(PROMPTS[:prompt_count]):
      for seed in seeds:
        for label,watermarked in ((0,False),(1,True)):
          torch.manual_seed(int(seed + pi*1009))
          encoded=tokenizer(prompt,return_tensors='pt')
          inputs={k:v.to(device) for k,v in encoded.items()}
          processors=[_SuppressEos(tokenizer.eos_token_id)]
          if watermarked:
            processors.insert(0,WatermarkLogitsProcessor(vocab=vocab,gamma=gamma,delta=delta,hash_key=key))
          with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=max_new_tokens,min_new_tokens=max_new_tokens,do_sample=True,
              temperature=.8,top_p=.9,repetition_penalty=1.15,no_repeat_ngram_size=3,pad_token_id=tokenizer.eos_token_id,eos_token_id=None,logits_processor=processors)
          continuation_ids=generated[0][inputs['input_ids'].shape[-1]:]
          text=tokenizer.decode(continuation_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
          anchored_ids=torch.cat((inputs['input_ids'][0,-1:].to(device),continuation_ids))
          scored=_safe_detect(detector,text,anchored_ids)
          sample_rows.append({"prompt_id":pi,"seed":seed,"label":label,"is_watermarked":label,
            "text":text,"generated_token_count":int(continuation_ids.numel()),"word_count":len(text.split()),"num_tokens_scored":scored['num_tokens_scored'],
            "num_green_tokens":scored['num_green_tokens'],"green_fraction":scored['green_fraction'],
            "z_score":scored['z_score'],"p_value":scored['p_value'],"prediction":int(bool(scored['prediction']))})
          if watermarked:
            for ai,attack in enumerate(attacks):
              attacked=apply_attack(text,attack,seed=seed+pi*100+ai)
              if attack == "none":
                attack_token_ids=anchored_ids
              else:
                attack_ids=tokenizer(attacked,return_tensors="pt",add_special_tokens=False)["input_ids"][0].to(device)
                attack_token_ids=torch.cat((inputs['input_ids'][0,-1:].to(device),attack_ids))
              result=_safe_detect(detector,attacked,attack_token_ids)
              attack_rows.append({"prompt_id":pi,"seed":seed,"attack":attack,"text":attacked,
                "word_count":len(attacked.split()),"num_tokens_scored":result['num_tokens_scored'],
                "green_fraction":result['green_fraction'],"z_score":result['z_score'],
                "p_value":result['p_value'],"prediction":int(bool(result['prediction']))})
    paired=_classification(sample_rows)
    positive=[r for r in sample_rows if r['label']==1]
    negative=[r for r in sample_rows if r['label']==0]
    attack_summary={}
    for attack in attacks:
      group=[r for r in attack_rows if r['attack']==attack]
      vals=[r['prediction'] for r in group]
      attack_summary[attack]={"n":len(group),"scorable_n":sum(r['num_tokens_scored']>0 for r in group),
        "detection_rate":float(np.mean(vals)) if vals else None,"detection_rate_95ci":_rate_ci(vals,seed=2026+len(attack))}
    def save_csv(path,rows):
      if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
          w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    samples_path=outdir/f'evaluation_samples_{run_id}.csv'; attacks_path=outdir/f'evaluation_attacks_{run_id}.csv'
    save_csv(samples_path,sample_rows);save_csv(attacks_path,attack_rows)
    report={"run_id":run_id,"created_utc":datetime.now(timezone.utc).isoformat(),"model_id":model_id,
      "device":str(device),"method":"Kirchenbauer et al. green-list watermark; simple_1 seeding scheme",
      "parameters":{"gamma":gamma,"delta":delta,"z_threshold":z_threshold,"max_new_tokens":max_new_tokens,"seeds":seeds,"prompt_count":prompt_count},
      "generation_settings":{"sampling":True,"temperature":0.8,"top_p":0.9,"repetition_penalty":1.15,"no_repeat_ngram_size":3,"fixed_length":True,"eos_suppressed":True},
      "detector_settings":{"ignore_repeated_bigrams":True,"score_original_generated_token_ids":True,"include_prompt_last_token_as_seed_anchor":True},
      "paired_control_vs_watermarked":paired,
      "watermarked_detection_rate_95ci":_rate_ci([r['prediction'] for r in positive]),
      "control_false_positive_rate_95ci":_rate_ci([r['prediction'] for r in negative],seed=2027),
      "attacks":attack_summary,"sample_count":len(sample_rows),"attack_sample_count":len(attack_rows),
      "artifacts":{"samples_csv":str(samples_path),"attacks_csv":str(attacks_path)},
      "interval_method":"Wilson score 95% confidence intervals for binary rates.",
      "limitations":["Small exploratory CPU evaluation; increase prompts and seeds for stable estimates.",
        "Word deletion, substitution and lexical paraphrase proxy are synthetic attacks, not human or neural semantic paraphrasing.",
        "Confidence intervals use bootstrap resampling of generated samples; prompt/seed observations may be correlated.",
        "Detection is statistical evidence for this scheme/key, not proof of authorship."]}
    report_path=outdir/f'evaluation_report_{run_id}.json';report['artifacts']['report_json']=str(report_path)
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
