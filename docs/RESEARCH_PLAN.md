# Research plan

## Research question
How accurately and robustly can a keyed statistical watermark be detected in LLM-generated text while preserving text quality?

## Baseline
Reproduce the soft green-list watermark using the authors' extended implementation. Start with gamma=0.25, delta=2.0 and the documented `selfhash` scheme. The detector must use matching watermark settings, tokenizer and key. Use repeated n-gram filtering as recommended by the authors.

## Experimental groups
1. Watermarked generations.
2. Unwatermarked generations from the same model and prompt set.
3. Human-written text, where a suitably licensed and documented dataset is available.

## Evaluation
- Detection: true-positive rate, false-positive rate, precision, recall, F1, ROC-AUC, z-score and p-value.
- Length: stratify results by token count.
- Quality: perplexity (with caveats), semantic similarity and blinded human ratings where feasible.
- Robustness: controlled deletion, substitution, paraphrase and sentence-reordering attacks.
- Statistics: report sample sizes, confidence intervals and threshold-selection procedure.

## Reproducibility record
For each run save model/tokenizer identifiers and revisions, package versions, prompt IDs, generation parameters, watermark parameters, key identifier (not the key), random seeds, raw scores and timestamp.

## Proposed extension (to be selected after baseline)
Choose one modification only after measuring baseline weaknesses. Candidate: tune watermark strength adaptively to token uncertainty, then compare the detection/quality trade-off against fixed-delta watermarking on held-out prompts.
