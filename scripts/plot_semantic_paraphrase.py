#!/usr/bin/env python3
"""Plot semantic paraphrase robustness and semantic-preservation diagnostics."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('csv',help='semantic_paraphrase_*.csv')
p.add_argument('--output-dir',default=None)
a=p.parse_args()
d=pd.read_csv(a.csv)
d=d[d.paraphrased_text.fillna('').str.strip().ne('')].copy()
out=Path(a.output_dir) if a.output_dir else Path(a.csv).parent/'figures'/'semantic_paraphrase'
out.mkdir(parents=True,exist_ok=True)

fig,ax=plt.subplots(figsize=(8,5))
ax.hist(d.semantic_cosine_similarity,bins=18)
ax.set_xlabel('MiniLM cosine similarity (original vs paraphrase)')
ax.set_ylabel('Number of samples')
ax.set_title('Semantic similarity of model-generated paraphrases')
fig.tight_layout(); fig.savefig(out/'semantic_similarity_distribution.png',dpi=180); plt.close(fig)

rates=[d.original_detected.mean(),d.paraphrase_detected.mean()]
fig,ax=plt.subplots(figsize=(7,5))
ax.bar(['Original watermarked text','After T5 paraphrasing'],rates)
ax.set_ylim(0,1); ax.set_ylabel('Detection rate (z ≥ 4)')
ax.set_title('Watermark detection before and after semantic paraphrasing')
for i,v in enumerate(rates): ax.text(i,v+0.02,f'{v:.1%}',ha='center')
fig.tight_layout(); fig.savefig(out/'detection_rate_comparison.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
sc=ax.scatter(d.semantic_cosine_similarity,d.paraphrase_z_score,c=d.paraphrase_detected,alpha=.75)
ax.axhline(4,linestyle='--',linewidth=1)
ax.set_xlabel('MiniLM cosine similarity')
ax.set_ylabel('Watermark detector z-score after paraphrasing')
ax.set_title('Semantic similarity vs residual watermark evidence')
fig.tight_layout(); fig.savefig(out/'similarity_vs_watermark_score.png',dpi=180); plt.close(fig)
print(f'Saved plots to {out}')
