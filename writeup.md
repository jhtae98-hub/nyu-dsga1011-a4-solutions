# Assignment 4: Fine-tuning Language Models

## Q1 — Results

The completed Q1 runs produced the following measured accuracies:

| Evaluation | Accuracy as a percentage |
| --- | --- |
| Debug evaluation | 89.2% |
| Full-test evaluation | 93.072% |


## Q2a — Transformation Design

I will apply **QWERTY neighbor-key typos** to simulate typing mistakes. I will give each eligible word occurrence a **50% chance** of receiving one typo, replacing an interior letter with a neighboring key.

### Procedure

1. **Identify eligible words.** I will scan each original review from left to right using:
   ```python
   r"<[^>]*>|\b\w+(?:['\u2019]\w+)*\b"
   ```
   I will skip HTML-like tags and consider only spans fully matching `[A-Za-z]{5,}`, meaning words containing at least five English letters. I will exclude these negation or qualification words regardless of capitalization:
   ```text
   cannot, neither, never, no, nobody, none,
   nor, not, nothing, nowhere, rarely, without
   ```
   I will leave short words, contractions, numbers, and words containing non-ASCII characters unchanged.

2. **Apply one substitution per selected word.** I use the existing `random.seed(0)` setup without reseeding between reviews. For each eligible occurrence, I will draw `random.random()` and select it when the value is below `0.50`. I will immediately choose an interior position using `random.randrange(1, len(word) - 1)` and replace that letter with an immediate horizontal neighbor from:
   ```text
   qwertyuiop
   asdfghjkl
   zxcvbnm
   ```
   I will list candidates left neighbor first, then right, omitting missing neighbors, and call `random.choice` once. I will preserve capitalization and consume no random draws for ineligible spans.

3. **Return the transformed example.** I will update only `text`, the review string, in a shallow copy of the example and preserve its label and other fields. I will keep punctuation, whitespace, tags, word lengths, and word order unchanged. I will apply the transformation once to the original review.

### Rationale

I chose nearby-key typos because they are plausible in quickly written reviews. By preserving word boundaries, first and last letters, contractions, and common negation cues, I expect to retain the intended sentiment in most cases.

## Q2b — Results

I evaluated my existing Q1 model on the full transformed test set without retraining it.

| Evaluation | Accuracy | Decrease from original |
| --- | ---: | ---: |
| Original test set | 93.072% | — |
| Initial transformation: 30% typo probability | 90.464% | 2.608 percentage points |
| Final transformation: 50% typo probability | 87.612% | 5.460 percentage points |

I increased the typo probability after the initial 30% trial produced less than a four-point decrease. My final setting is **50%**, which produced a **5.460 percentage-point decrease**, exceeding Q2b's required decrease of more than four points.

## Q3 — Report and Analysis

| Test set | Before augmentation | After augmentation | Change |
|---|---:|---:|---:|
| Original | 93.072% | 92.784% | −0.288 percentage points |
| Transformed | 87.612% | 89.348% | +1.736 percentage points |

The model’s performance on the transformed test data improved after augmentation, while accuracy on the original test set decreased slightly.

Adding more typo-containing training examples most likely helped the model recognize sentiment despite spelling mistakes. However, the added noise may also have shifted the model’s learned patterns slightly, which explains the small decrease on clean reviews.

One limitation is that my augmentation handles only QWERTY neighbor-key typos. So, the improvement may not extend to other OOD changes, such as replacing words with synonyms.

## Q4 — Data Statistics and Processing

### Table 1 — Before preprocessing

| Statistics Name | Train | Dev |
| --- | ---: | ---: |
| Number of examples | 4225 | 466 |
| Mean sentence length | 17.10 | 17.07 |
| Mean SQL query length | 216.37 | 210.05 |
| Vocabulary size (natural language) | 791 | 465 |
| Vocabulary size (SQL) | 555 | 395 |

### Table 2 — After preprocessing

| Statistics Name | Train | Dev |
| --- | ---: | ---: |
| Number of examples | 4225 | 466 |
| Mean sentence length | 22.10 | 22.07 |
| Mean SQL query length | 216.21 | 209.89 |
| Vocabulary size (natural language) | 795 | 469 |
| Vocabulary size (SQL) | 556 | 396 |
