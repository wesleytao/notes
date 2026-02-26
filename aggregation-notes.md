# NLP Evaluation Metrics – Study Notes

---

## 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

### What It Is

ROUGE is a family of **automatic, lexical (surface-form) metrics** that measure how much a generated text (candidate) overlaps with one or more human reference texts. It is **syntactic/surface-based, not semantic**.

> **One-liner:** ROUGE measures how much a generated text overlaps with reference texts by counting shared words, phrases, or token sequences.

### Core Question It Answers

*"How much of the reference did my generated text cover?"*

---

### Common Variants

| Variant | What It Measures | Sensitivity |
|---------|-----------------|-------------|
| **ROUGE-1** | Unigram (single word) overlap | Broad content coverage |
| **ROUGE-2** | Bigram (2-word sequence) overlap | Local phrasing / fluency |
| **ROUGE-L** | Longest Common Subsequence (LCS) | Token order / structure |
| ROUGE-S/SU | Skip-bigram overlap | Older, less common |
| ROUGE-W | Weighted LCS | Rare |

---

### How It's Calculated

#### ROUGE-N (n-gram overlap)

Let $G_n(x)$ = multiset of all n-grams in text $x$, and $c_x(g)$ = count of n-gram $g$ in $x$.

**Overlap count:**

$$\text{overlap}_n(C, R) = \sum_{g \in G_n(R)} \min(c_C(g),\ c_R(g))$$

**Recall / Precision / F1:**

$$\text{Recall} = \frac{\text{overlap}_n(C, R)}{\sum_{g \in G_n(R)} c_R(g)}$$

$$\text{Precision} = \frac{\text{overlap}_n(C, R)}{\sum_{g \in G_n(C)} c_C(g)}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### ROUGE-L (Longest Common Subsequence)

$$P_L = \frac{\text{LCS}(C, R)}{|C|}, \quad R_L = \frac{\text{LCS}(C, R)}{|R|}$$

$$F_L = \frac{(1 + \beta^2) \cdot P_L \cdot R_L}{R_L + \beta^2 \cdot P_L}$$

($\beta$ weights recall vs. precision; many toolkits default to emphasizing recall.)

---

### Worked Example (ROUGE-1)

- **Candidate:** "the cat is on the mat"
- **Reference:** "there is a cat on the mat"

Shared unigrams: *is, cat, on, the, mat* → overlap = **5**

| Metric | Calculation | Value |
|--------|------------|-------|
| Recall | 5 / 7 | ≈ 0.714 |
| Precision | 5 / 6 | ≈ 0.833 |
| F1 | 2 × (5/7) × (5/6) / ((5/7) + (5/6)) | ≈ 0.769 |

---

### Handling Duplicates in ROUGE

Duplicates are handled by the `min` in the overlap formula. Extra repeats in the candidate **do not increase overlap beyond the reference count**, but they **hurt precision** (because candidate length grows).

- Candidate: `cat:3`, Reference: `cat:2` → overlap contribution = min(3, 2) = **2**
- The third "cat" in the candidate adds nothing to overlap but inflates the denominator for precision.

### Preprocessing: Stemming vs. Lemmatization

Many ROUGE implementations support **stemming** (e.g., Porter stemmer), which helps match suffixes: `cats → cat`, `taking → take`.

However, stemming typically **does not** normalize irregular forms: `took` stays `took` (not `take`), `taken` stays `taken`.

If you need `taken / took / take` treated as the same token, you need **lemmatization or custom normalization** applied *before* computing ROUGE.

---

### Best Use Cases (ROUGE)

- Summarization with reference summaries (news, reports, meeting minutes)
- Model comparison within the same dataset/setup
- Ablations and monitoring — cheap signal to catch regressions

### Drawbacks

- **Penalizes paraphrases** — same meaning, different words → low score
- **Doesn't measure factuality** — hallucinations may not hurt if overlap is still high
- **Sensitive to preprocessing** — tokenization, stemming, sentence splitting change scores
- **Reference-dependent** — narrow/few references make it unfair
- **Rewards copying** — extractive summaries score higher

### When to Pick Which Variant

- **ROUGE-1** → when you mostly care about content coverage and wording can vary
- **ROUGE-2** → when local phrasing fidelity matters
- **ROUGE-L** → when ordering/structure of key tokens matters

Most papers report **ROUGE-1, ROUGE-2, and ROUGE-L (F1)** together.

---

### Python Code

```python
# Option A: Hugging Face evaluate
# pip install evaluate rouge_score
import evaluate

rouge = evaluate.load("rouge")
preds = ["the cat sat on the mat"]
refs  = ["there is a cat on the mat"]

scores = rouge.compute(
    predictions=preds,
    references=refs,
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)
print(scores)

# Option B: rouge_score (explicit control)
# pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score("there is a cat on the mat", "the cat sat on the mat")
for k, v in scores.items():
    print(k, "P:", v.precision, "R:", v.recall, "F1:", v.fmeasure)
```

---

## 2. BLEU (Bilingual Evaluation Understudy)

### What It Is

BLEU is a **precision-oriented, lexical (surface-form) metric** originally designed for **machine translation**. It measures how many n-grams in the candidate appear in the reference(s).

> **One-liner:** BLEU measures how much of the generated text's wording matches the reference, with a penalty for being too short.

### Core Question It Answers

*"How much of what I generated actually appears in the reference?"* (precision-focused)

---

### How It's Calculated

#### Step 1: Modified n-gram Precision

For each n-gram in the candidate, count how many times it appears — but **clip** each count to the maximum count in any single reference (prevents gaming by repeating words).

$$p_n = \frac{\sum_{g \in G_n(C)} \min\left(c_C(g),\ \max_{R_j} c_{R_j}(g)\right)}{\sum_{g \in G_n(C)} c_C(g)}$$

#### Step 2: Brevity Penalty (BP)

BLEU penalizes candidates that are shorter than the reference to prevent high precision from very short outputs.

$$\text{BP} = \begin{cases} 1 & \text{if } |C| \geq |R| \\ e^{1 - |R|/|C|} & \text{if } |C| < |R| \end{cases}$$

#### Step 3: Final BLEU Score

Combine modified precisions for n = 1, 2, 3, 4 (standard BLEU-4) using a geometric mean, weighted equally:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \cdot \ln(p_n)\right)$$

where $w_n = 1/N$ (typically $N = 4$, so each weight = 0.25).

---

### Worked Example (BLEU — simplified with unigrams only)

- **Candidate:** "the cat is on the mat"
- **Reference:** "there is a cat on the mat"

Candidate unigrams: {the: 2, cat: 1, is: 1, on: 1, mat: 1}
Clipped counts against reference: {the: 1, cat: 1, is: 1, on: 1, mat: 1} → clipped sum = **5**

$$p_1 = \frac{5}{6} \approx 0.833$$

Lengths: |C| = 6, |R| = 7 → C is shorter, so BP = $e^{1 - 7/6}$ ≈ 0.846

BLEU-1 (unigram only) ≈ 0.846 × 0.833 ≈ **0.705**

*(Full BLEU-4 would also use bigram, trigram, and 4-gram precisions in the geometric mean.)*

---

### Corpus-Level vs. Sentence-Level

BLEU was designed as a **corpus-level** metric (aggregate over many sentence pairs). Sentence-level BLEU can be very noisy — a single sentence with zero 4-gram matches gives BLEU = 0. Smoothing methods (e.g., `smoothing_function` in NLTK) help for sentence-level use.

---

### Best Use Cases

- **Machine translation** — the metric BLEU was built for; the standard MT benchmark metric for decades
- Corpus-level system comparison (ranking MT models)
- Any generation task where precision of wording matters

### Drawbacks

- **Precision-only** — doesn't penalize missing important content (low recall awareness)
- **Penalizes paraphrases** — same meaning, different wording → low score
- **Doesn't measure meaning or factuality** — purely lexical
- **Noisy at sentence level** — best used at corpus level
- **Brevity penalty is a blunt instrument** — doesn't distinguish between truncated vs. concise outputs
- **Sensitive to tokenization/casing** — preprocessing choices change scores

---

### Python Code

```python
# Option A: NLTK (sentence-level with smoothing)
# pip install nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["there", "is", "a", "cat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]

# Sentence-level BLEU-4 with smoothing
smooth = SmoothingFunction().method1
score = sentence_bleu(reference, candidate, smoothing_function=smooth)
print("BLEU (sentence):", score)

# Option B: sacrebleu (corpus-level, standardized)
# pip install sacrebleu
import sacrebleu

refs = [["there is a cat on the mat"]]   # list of list of references
preds = ["the cat is on the mat"]

bleu = sacrebleu.corpus_bleu(preds, refs)
print("BLEU (corpus):", bleu.score)
print(bleu)  # detailed breakdown

# Option C: Hugging Face evaluate
# pip install evaluate
import evaluate

bleu_metric = evaluate.load("bleu")
preds = ["the cat is on the mat"]
refs  = [["there is a cat on the mat"]]
result = bleu_metric.compute(predictions=preds, references=refs)
print("BLEU:", result["bleu"])
print("Precisions:", result["precisions"])  # p1, p2, p3, p4
print("BP:", result["brevity_penalty"])
```

---

### BLEU vs. ROUGE — Head-to-Head Comparison

| Dimension | BLEU | ROUGE |
|-----------|------|-------|
| **Full name** | Bilingual Evaluation Understudy | Recall-Oriented Understudy for Gisting Evaluation |
| **Orientation** | **Precision-focused** — "How much of my output is in the reference?" | **Recall-focused** — "How much of the reference is in my output?" |
| **Primary domain** | Machine translation | Summarization |
| **Basis** | Syntactic (n-gram) | Syntactic (n-gram + LCS) |
| **N-gram default** | BLEU-4 (1- through 4-grams, geometric mean) | ROUGE-1, ROUGE-2, ROUGE-L reported separately |
| **Aggregation** | Geometric mean of $p_1 \ldots p_4$ | Typically F1 per variant |
| **Short-output handling** | Brevity Penalty (explicit) | Handled implicitly via precision in F1 |
| **Granularity** | Best at corpus level; noisy per-sentence | Works at both sentence and corpus level |
| **Multiple references** | Clips n-gram counts to max across references | Typically takes max score across references |
| **Reports** | Single combined score | Separate scores per variant (ROUGE-1, -2, -L) |
| **Captures meaning?** | No | No |
| **Speed** | Very fast | Very fast |

#### When to Use Which

- **Use BLEU** when you care about **precision of wording** — did the system generate text that actually appears in the reference? Best for **machine translation**.
- **Use ROUGE** when you care about **coverage/recall** — did the system capture the important content from the reference? Best for **summarization**.
- **Use both** when evaluating a system that needs to be both precise and comprehensive (e.g., abstractive summarization, data-to-text).

#### Key Intuition

> **BLEU asks:** "Of the words I generated, how many are correct?"
> **ROUGE asks:** "Of the words that should appear, how many did I generate?"

A **verbose, rambling** output might score high on ROUGE (it covers everything) but low on BLEU (lots of extra junk drags down precision). A **very short, cherry-picked** output might score high on BLEU (everything it says is correct) but low on ROUGE (it missed most of the reference content).

---

### Deep Dive: Multi-Reference Handling (BLEU vs. ROUGE)

This is a subtle but important difference that often trips people up.

#### BLEU: Per-n-gram max clipping across references

BLEU handles multiple references **simultaneously at the n-gram level**. For each n-gram, it takes the **max count across all references** as the clip cap. This means **different n-grams can be justified by different references** in a single score computation.

$$\text{clip}(g) = \min\left(c_C(g),\ \max_j c_{R_j}(g)\right)$$

BLEU can "mix-and-match" evidence: one reference might justify "the" appearing twice, while another justifies "cat" appearing once — both contribute to the same precision score.

#### ROUGE: Pairwise scores then aggregate

ROUGE (in most implementations) computes a **separate score against each reference**, then reduces:

$$\text{ROUGE}_{best} = \max_j \text{ROUGE}(C, R_j) \quad \text{or} \quad \text{ROUGE}_{avg} = \frac{1}{k}\sum_j \text{ROUGE}(C, R_j)$$

Each pairwise score is limited to what **one reference** can support. There is no cross-reference mixing at the n-gram level.

#### Worked Example: Multi-Reference Mix-and-Match

**Candidate** $C$: `"the the cat"`

**References:**
- $R_1$: `"the cat"` (supports `cat`)
- $R_2$: `"the the dog"` (supports second `the`)

**BLEU-1 (unigram precision):**
- `"the"`: candidate count = 2, max across refs = max(1, 2) = 2 → clipped = min(2, 2) = **2** ✓ (from $R_2$)
- `"cat"`: candidate count = 1, max across refs = max(1, 0) = 1 → clipped = min(1, 1) = **1** ✓ (from $R_1$)
- $p_1$ = (2 + 1) / 3 = **1.0** — perfect precision by combining evidence from both references

**Pairwise ROUGE-1 precision:**
- vs $R_1$ (`"the cat"`): overlap = min(2,1) + min(1,1) = 1 + 1 = 2 → precision = 2/3 ≈ 0.667
- vs $R_2$ (`"the the dog"`): overlap = min(2,2) + min(1,0) = 2 + 0 = 2 → precision = 2/3 ≈ 0.667
- Best pairwise = **0.667** — neither single reference can support everything

**Takeaway:** BLEU achieved 1.0 precision by mixing evidence across references, while ROUGE's best pairwise score was only 0.667. This difference matters when references are complementary rather than overlapping.

#### Key Insight: Single Reference Equivalence

For a **single reference**, BLEU's modified precision $p_n$ is essentially the same formula as **ROUGE-N precision** — both use `min(candidate_count, reference_count)` in the numerator and candidate n-gram count in the denominator. The differences only emerge with multiple references and with how the final score is assembled (BP + geometric mean vs. separate F1 scores).

---

### Reporting Defaults (Practical)

- **ROUGE** is often reported as **F1** nowadays (though it was originally recall-oriented)
- **BLEU** is typically reported as a **single corpus-level score** (not P/R/F1), built from modified precisions + BP

---

## 3. BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)

### What It Is

BLEURT is a **learned, reference-based, semantic metric**. It encodes (reference, candidate) together with a Transformer and predicts a scalar quality score trained to correlate with human judgments.

### How It Works

1. Encode `[reference ; candidate]` with a Transformer (BERT / RemBERT)
2. A regression head predicts a single scalar score
3. Trained in phases: pretrained LM → synthetic pretraining → fine-tuned on human ratings (e.g., WMT)

### "Formula"

$$\text{BLEURT}(r, c) = f_\theta(\text{Encoder}([r; c]))$$

Training objective:

$$\min_\theta \sum_i \left( f_\theta(r_i, c_i) - y_i \right)^2$$

where $y_i$ are human quality ratings.

### Interpreting the Score

- Scores depend on the checkpoint (don't compare across different BLEURT models)
- BLEURT-20 checkpoint: scores roughly 0–1 but can exceed that range
- Average over a dataset for stability

### Best Use Cases

- Machine translation / summarization / NLG with references
- System ranking and regression testing when you need meaning-awareness beyond n-gram overlap

### Drawbacks

- **Still reference-based** — unfair with incomplete/low-quality references
- **Doesn't check faithfulness** — judges similarity to reference, not factual grounding
- **Heavier & slower** than ROUGE (runs a Transformer; TensorFlow-based)
- **Domain shift** — learned model can behave oddly on out-of-distribution styles

### Python Code

```python
# Hugging Face evaluate
# pip install evaluate tensorflow
import evaluate

bleurt = evaluate.load("bleurt", "BLEURT-20")
preds = ["the cat sat on the mat"]
refs  = ["there is a cat on the mat"]
out = bleurt.compute(predictions=preds, references=refs)
print(out["scores"])

# Official BLEURT package
from bleurt import score
scorer = score.BleurtScorer("BLEURT-20")
scores = scorer.score(
    references=["there is a cat on the mat"],
    candidates=["the cat sat on the mat"]
)
print(scores)
```

---

## 4. STS Score (Semantic Textual Similarity)

### What It Is

A **semantic similarity score** between two texts (typically 0–1 or 0–5 depending on model). It captures whether two texts mean the same thing, regardless of wording.

### Two Common Approaches

| Approach | Speed | Accuracy | How |
|----------|-------|----------|-----|
| **Bi-encoder** + cosine | Fast | Good | Embed each text independently, compute cosine similarity |
| **Cross-encoder** | Slow | Better | Concatenate pair, predict similarity directly |

### Formula (Bi-encoder)

$$\text{STS}(x, y) = \cos(E(x), E(y)) = \frac{E(x) \cdot E(y)}{\|E(x)\| \cdot \|E(y)\|}$$

### Best Use Cases

- Paraphrase / near-duplicate detection and clustering
- Reference-based generation eval when meaning matters more than exact wording

### Drawbacks

- **Not a factuality check** — semantically similar but wrong statements can score high
- **Domain drift** can skew scores
- Cross-encoders are compute-heavy

### Python Code

```python
# pip install -U sentence-transformers
from sentence_transformers import SentenceTransformer, CrossEncoder, util

a = "A plane is taking off."
b = "An airplane is taking off."

# Bi-encoder (fast)
bi = SentenceTransformer("all-MiniLM-L6-v2")
ea = bi.encode(a, convert_to_tensor=True, normalize_embeddings=True)
eb = bi.encode(b, convert_to_tensor=True, normalize_embeddings=True)
sts_cos = util.dot_score(ea, eb).item()
print("STS (bi-encoder):", sts_cos)

# Cross-encoder (more accurate)
ce = CrossEncoder("cross-encoder/stsb-roberta-large")
sts_ce = float(ce.predict([(a, b)])[0])
print("STS (cross-encoder):", sts_ce)
```

---

## 5. SAS Score (Semantic Answer Similarity)

### What It Is

A **semantic similarity metric specialized for QA evaluation**. Compares a predicted answer to ground-truth answer(s) using a cross-encoder, giving credit for semantically equivalent answers even with zero lexical overlap.

### Formula (Multi-reference)

$$\text{SAS}(\hat{a}, \{a_i\}) = \max_{i=1..k} f_\theta(\hat{a}, a_i)$$

where $f_\theta$ is a cross-encoder STS scorer returning similarity in [0, 1].

### Best Use Cases

- Extractive / short-form QA where Exact Match (EM) / F1 under-counts correct paraphrases
- Datasets with multiple valid answers or incomplete references

### Drawbacks

- Can **over-reward vague answers** (e.g., "somewhere in Europe") if they're close in embedding space
- Doesn't guarantee **groundedness** in the source context
- Cross-encoder can be slow at scale

### Python Code

```python
from sentence_transformers import CrossEncoder

pred = "Barack Obama"
refs = ["Obama", "President Barack Obama", "Barack Hussein Obama II"]

sas_model = CrossEncoder("cross-encoder/stsb-roberta-large")
pairs = [(pred, r) for r in refs]
scores = sas_model.predict(pairs)

sas = float(max(scores))
print("SAS:", sas)
print("Per-ref:", list(map(float, scores)))
```

---

## 6. Quick Comparison Table

| Metric | Type | Basis | Orientation | Reference Needed? | Captures Meaning? | Speed |
|--------|------|-------|-------------|-------------------|-------------------|-------|
| **BLEU** | Lexical overlap | Syntactic (n-grams) | Precision | Yes | No | Very fast |
| **ROUGE** | Lexical overlap | Syntactic (n-grams / LCS) | Recall | Yes | No | Very fast |
| **BLEURT** | Learned regression | Semantic (Transformer) | — | Yes | Yes | Slow |
| **STS** | Embedding similarity | Semantic (bi/cross-encoder) | — | Yes (pairwise) | Yes | Medium–Slow |
| **SAS** | Cross-encoder similarity | Semantic (QA-tuned) | — | Yes (answer-level) | Yes | Slow |

---

## 7. Key Takeaways for Exam / Review

1. **BLEU is precision-oriented and syntactic** — best for machine translation. Uses a brevity penalty instead of recall. Best at corpus level.
2. **ROUGE is recall-oriented and syntactic** — best for summarization. Counts n-gram/LCS overlaps. Reports F1 per variant.
3. **BLEU vs. ROUGE in one line:** BLEU asks "is what I said correct?"; ROUGE asks "did I say everything important?"
4. **Multi-reference handling differs critically:** BLEU mixes evidence across references at the n-gram level (per-n-gram max clipping). ROUGE computes pairwise scores then aggregates (max or average). For a single reference, their precision formulas are equivalent.
5. **Duplicates:** Both use `min(candidate_count, reference_count)` — repeating words doesn't help beyond the reference count but hurts precision.
6. **Stemming ≠ lemmatization:** ROUGE stemming handles suffixes (`cats → cat`) but not irregular forms (`took ≠ take`). Use lemmatization before ROUGE if you need that.
7. **BLEURT is semantic and learned** — trained on human judgments, better at capturing meaning, but heavier.
8. **STS** gives a general-purpose semantic similarity score. Bi-encoders are fast; cross-encoders are more accurate.
9. **SAS** is STS specialized for QA — uses max-over-references to handle multiple valid answers.
10. **None of these measure factuality/faithfulness** — a hallucinated but reference-similar sentence can still score well. Pair with faithfulness checks (e.g., NLI-based) for production.
11. **Always compare systems using the same metric settings** (same variant, preprocessing, checkpoint).
12. **Best practice:** Use BLEU/ROUGE as fast regression signals, and supplement with semantic metrics (BLEURT/STS/SAS) + faithfulness checks for a complete picture.

---

## 8. Mental Checklist

- Care about **coverage of reference content**? → ROUGE (recall / F1)
- Care about **how much of your output is supported by the reference**? → BLEU (precision)
- Need **meaning-aware** evaluation? → BLEURT, STS, or SAS
- Need **factuality / faithfulness**? → None of the above alone — add NLI-based or QA-based faithfulness metrics
- Both BLEU and ROUGE are **surface overlap** — they can miss meaning, paraphrases, and factuality

---

*Study notes compiled from NLP evaluation metric fundamentals.*
