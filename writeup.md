# Assignment 4: Fine-tuning Language Models

## Q1 — Results

The completed Q1 runs produced the following measured accuracies:

| Evaluation | Accuracy as a percentage |
| --- | --- |
| Debug evaluation | 89.2% |
| Full-test evaluation | 93.072% |


## Q2a — Transformation Design

I will apply **QWERTY neighbor-key typos** to simulate typing mistakes, and give each eligible word occurrence a **50% chance** of receiving one typo, replacing an interior letter with a neighboring key.

### Procedure

1. **Identify eligible words.** I will scan each original review from left to right using:
   ```python
   r"<[^>]*>|\b\w+(?:['\u2019]\w+)*\b"
   ```
   Then, I will skip HTML-like tags and consider only spans fully matching `[A-Za-z]{5,}`, meaning words containing at least five English letters. These negation or qualification words will be excluded regardless of capitalization:
   ```text
   cannot, neither, never, no, nobody, none,
   nor, not, nothing, nowhere, rarely, without
   ```
   And I will leave short words, contractions, numbers, and words containing non-ASCII characters unchanged.

2. **Apply one substitution per selected word.** I use the existing `random.seed(0)` setup without reseeding between reviews. For each eligible occurrence, I will draw `random.random()` and select it when the value is below `0.50`. Then, I'll choose an interior position using `random.randrange(1, len(word) - 1)` and replace that letter with an immediate horizontal neighbor from:
   ```text
   qwertyuiop
   asdfghjkl
   zxcvbnm
   ```
   After that, I will list candidates left neighbor first, then right, omitting missing neighbors, and call `random.choice` once, preserving capitalization.

3. **Return the transformed example.** I will update only `text`, the review string, in a shallow copy of the example and preserve its label and other fields. Punctuation, whitespace, tags, word lengths, and word order will be left unchanged, and the transformation will be applied once to the original review.

### Rationale

I chose nearby-key typos because they are plausible in quickly written reviews. By preserving word boundaries, first and last letters, contractions, and common negation cues, I expect to retain the intended sentiment in most cases.

## Q2b — Results

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

## Q5 — T5 Fine-tuning Details

**Table 3: Best-performing T5 model configuration**

| Design choice | Description |
|---|---|
| Data processing | I removed surrounding whitespace from questions and SQL queries, and prefixed each question with `translate English to SQL: `. I preserved capitalization, punctuation, internal whitespace, and question–SQL alignment. |
| Tokenization | I used `T5TokenizerFast` from `google-t5/t5-small` for both questions and SQL. I added `<` and `<=` as ordinary tokens to preserve these SQL operators. I also appended EOS to each question and SQL target and set their length limits to 128 and 512 tokens, including EOS. In addition, I padded each batch dynamically, masked padding, and used `-100` for padded target labels to exclude them from loss. Finally, I formed decoder inputs by shifting the targets right and starting with PAD. |
| Architecture | I loaded `google-t5/t5-small` using `T5ForConditionalGeneration` and resized its embeddings to 32,102 tokens. Then, I fine-tuned all parameters, including the encoder, decoder, shared embeddings, and tied output head.|
| Hyperparameters | I used AdamW with learning rate `3e-4`, weight decay `0`, training/evaluation batch sizes of `16`, and seed `42`. Moreover, I used a linear learning-rate schedule with one warmup epoch and capped training at 15 epochs, with early stopping after 3 consecutive epochs without improvement in development Record F1. I also generated SQL using 4 beams, no sampling, and at most 512 new tokens. |

## Q6 — Results and Error Analysis

**Table 4: Development and test results (%)**

| System                              | Query EM | Record EM | F1 score |
| ----------------------------------- | -------: | --------: | -------: |
| **Dev Results**                     |          |           |          |
| Full model, best epoch 13 of 15     |    55.58 |     72.75 |    78.76 |
| Baseline, 5 epochs, greedy decoding |    28.76 |     45.92 |    54.32 |
| Same baseline checkpoint, 4 beams   |    29.40 |     45.71 |    55.27 |
| **Test Results**                    |          |           |          |
| T5 fine-tuning                      |     0.00 |     71.99 |    76.60 |

I used a 10-second per-query SQL timeout and two workers for all development comparisons. Four-beam decoding slightly improved baseline F1, while the longer 15 epoch training run achieved substantially higher development F1. I selected its epoch-13 checkpoint.

The test Query EM of 0.00 is reproduced exactly as displayed by Gradescope.

**Table 5: Qualitative analysis on the development set**

| Error Type                         | Example Of Error                                                                                                                  | Error Description                                                                                                                                                    | Statistics |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| Incorrect or missing time filters  | For “after 7am,” it generated `departure_time > 1800` instead of `departure_time > 700`.                                          | Executable queries used incorrect time values, operators, or omitted constraints. This suggests difficulty translating time expressions into precise SQL conditions. |     36/466 |
| SQL syntax errors                  | For an aircraft query with ‘before 10am,’ the prediction ended with `flight_1.departure_time< 1000 ) )`, containing one extra closing parenthesis. | I found malformed or incomplete SQL, which may reflect difficulty maintaining nested query structure during generation.                                              |     17/466 |
| Invalid schema or alias references | For airport-to-downtown transportation in Pittsburgh, it referenced `city_2.city_code` without declaring `city city_2` in `FROM`. | I found 15 missing or invalid references and 3 ambiguous references. These suggest difficulty consistently tracking tables and aliases.                              |     18/466 |

## Q7 — Performance Evaluation

I submitted the final test SQL queries and their associated database records. My model achieved a test Record F1 of **76.60**, earning **25/25** performance points.

## Submission Links

- [GitHub repository](https://github.com/jhtae98-hub/nyu-dsga1011-a4-solutions)
- [Google Drive folder](https://drive.google.com/drive/folders/1MdU0cuqHqmyIYim0LcTNSl89lX1Z5_rF?usp=drive_link)
