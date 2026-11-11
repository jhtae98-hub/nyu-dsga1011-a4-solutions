"""Measure train/dev text-to-SQL statistics using only the T5 tokenizer.

Run from the repository root: python part2/data_stats.py
Or from part2: python data_stats.py
"""

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from preprocessing import QUESTION_PREFIX, preprocess_question, preprocess_sql


CHECKPOINT = "google-t5/t5-small"
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "data_stats_results.json"
STATISTICS = (
    ("Number of examples", "num_examples"),
    ("Mean question length (T5 tokens)", "mean_question_tokens"),
    ("Mean SQL query length (T5 tokens)", "mean_sql_tokens"),
    ("Vocabulary size (questions)", "question_vocabulary_size"),
    ("Vocabulary size (SQL)", "sql_vocabulary_size"),
)


def read_records(path: Path) -> list[str]:
    # Universal newline handling removes LF, CRLF, or CR separators, but no spaces.
    with path.open("r", encoding="utf-8", newline=None) as handle:
        records = [line.removesuffix("\n") for line in handle]
    if not records:
        raise ValueError(f"{path}: no records found")
    empty_lines = [str(index) for index, text in enumerate(records, 1) if not text.strip()]
    if empty_lines:
        raise ValueError(f"{path}: unexpected empty records at line(s) {', '.join(empty_lines)}")
    return records


def read_pairs(split: str) -> tuple[list[str], list[str]]:
    questions = read_records(DATA_DIR / f"{split}.nl")
    queries = read_records(DATA_DIR / f"{split}.sql")
    if len(questions) != len(queries):
        raise ValueError(
            f"{split}: question/SQL counts differ: {len(questions)} versus {len(queries)}"
        )
    return questions, queries


def measure_texts(texts: list[str], tokenizer, batch_size: int) -> dict:
    total_tokens = 0
    observed_ids = set()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start:start + batch_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        for token_ids in encoded["input_ids"]:
            total_tokens += len(token_ids)
            # Keep one vocabulary set across all batches in this split/field.
            observed_ids.update(token_ids)
    return {
        "total_tokens": total_tokens,
        "mean_tokens": total_tokens / len(texts),
        "distinct_token_ids": len(observed_ids),
    }


def measure_pairs(questions: list[str], queries: list[str], tokenizer, batch_size: int) -> dict:
    question_stats = measure_texts(questions, tokenizer, batch_size)
    sql_stats = measure_texts(queries, tokenizer, batch_size)
    return {
        "num_examples": len(questions),
        "mean_question_tokens": question_stats["mean_tokens"],
        "mean_sql_tokens": sql_stats["mean_tokens"],
        "question_vocabulary_size": question_stats["distinct_token_ids"],
        "sql_vocabulary_size": sql_stats["distinct_token_ids"],
        "total_question_tokens": question_stats["total_tokens"],
        "total_sql_tokens": sql_stats["total_tokens"],
    }


def format_table(results: dict) -> str:
    lines = ["| Statistic | Train | Dev |", "| --- | ---: | ---: |"]
    for label, key in STATISTICS:
        values = []
        for split in ("train", "dev"):
            value = results[split][key]
            values.append(f"{value:.2f}" if key.startswith("mean_") else str(value))
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    # Validate all pairs before loading the tokenizer; never read the test split.
    pairs = {split: read_pairs(split) for split in ("train", "dev")}
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import T5TokenizerFast

    print(f"Loading tokenizer only: {CHECKPOINT}", flush=True)
    tokenizer = T5TokenizerFast.from_pretrained(CHECKPOINT)
    before, after, whitespace_changes = {}, {}, {}
    for split, (questions, queries) in pairs.items():
        print(f"Measuring {split}: {len(questions)} aligned pairs", flush=True)
        before[split] = measure_pairs(questions, queries, tokenizer, args.batch_size)
        processed_questions = [preprocess_question(question) for question in questions]
        processed_queries = [preprocess_sql(sql) for sql in queries]
        after[split] = measure_pairs(processed_questions, processed_queries, tokenizer, args.batch_size)
        whitespace_changes[split] = {
            "questions_with_edge_whitespace": sum(question != question.strip() for question in questions),
            "sql_with_edge_whitespace": sum(sql != sql.strip() for sql in queries),
        }

    results = {
        "tokenizer_checkpoint": CHECKPOINT,
        "tokenizer_class": type(tokenizer).__name__,
        "versions": {
            "python": sys.version.split()[0],
            "transformers": version("transformers"),
            "tokenizers": version("tokenizers"),
        },
        "python_executable": sys.executable,
        "batch_size": args.batch_size,
        "tokenizer_settings": {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        },
        "preprocessing": {
            "question": '"translate English to SQL: " + question.strip()',
            "sql": "sql.strip()",
            "question_prefix": QUESTION_PREFIX,
            "preserve_pair_order": True,
        },
        "counting_convention": {
            "before": "Remove file line separators only; preserve all other line content.",
            "mean": "Total returned token IDs divided by the number of examples; no rounding in JSON.",
            "vocabulary": "Distinct returned token IDs across the entire split, separately for questions and SQL.",
            "after": "Question lengths and vocabulary include the task prefix.",
            "validation": "Reject mismatched counts or empty/whitespace-only records; never drop pairs.",
        },
        "input_files": {
            split: {"questions": f"data/{split}.nl", "sql": f"data/{split}.sql"}
            for split in pairs
        },
        "edge_whitespace_changes": whitespace_changes,
        "before": before,
        "after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nTable 1: Before preprocessing")
    print(format_table(before))
    print("\nTable 2: After preprocessing")
    print(format_table(after))
    print(f"\nSaved unrounded measurements: {args.output.resolve()}")


if __name__ == "__main__":
    main()
