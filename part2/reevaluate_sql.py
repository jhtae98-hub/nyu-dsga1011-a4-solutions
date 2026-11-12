"""Re-execute existing development SQL with independent per-query deadlines."""

import argparse
import hashlib
import io
import json
import math
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TIMEOUT_MESSAGE = "Query timed out"
PROGRESS_STEPS = 1000
FETCH_BATCH_SIZE = 1000


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted_sql", type=Path, required=True)
    parser.add_argument("--eval_name", required=True)
    parser.add_argument("--timeout_seconds", type=float, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--database", type=Path, default=SCRIPT_DIR / "data/flight_database.db")
    parser.add_argument("--reference_sql", type=Path, default=SCRIPT_DIR / "data/dev.sql")
    parser.add_argument("--reference_records", type=Path, default=SCRIPT_DIR / "records/ground_truth_dev.pkl")
    parser.add_argument("--results_root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--records_root", type=Path, default=SCRIPT_DIR / "records")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout_seconds must be finite and positive.")
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if (not args.eval_name or args.eval_name in (".", "..")
            or "/" in args.eval_name or "\\" in args.eval_name):
        parser.error("--eval_name must be a single nonempty path component.")
    return args


def read_sql(path):
    # Retain a byte-for-byte snapshot for export. Only line separators are removed
    # from the queries passed to SQLite; whitespace and SQL contents are unchanged.
    payload = Path(path).read_bytes()
    with io.StringIO(payload.decode("utf-8"), newline=None) as handle:
        queries = [line.removesuffix("\n") for line in handle]
    if not queries:
        raise ValueError(f"{path}: no SQL records found.")
    return payload, queries


def execute_query(query, database_uri, timeout_seconds):
    # Start the clock inside the worker, never while this query is queued.
    deadline = time.monotonic() + timeout_seconds
    interrupted_by_deadline = False
    connection = cursor = None

    def progress():
        nonlocal interrupted_by_deadline
        if time.monotonic() >= deadline:
            interrupted_by_deadline = True
            return 1
        return 0

    try:
        # Do not permit lock waiting outside SQLite's progress-handler checks.
        connection = sqlite3.connect(database_uri, uri=True, timeout=0)
        connection.set_progress_handler(progress, PROGRESS_STEPS)
        cursor = connection.cursor()
        cursor.execute("PRAGMA query_only = ON")
        if time.monotonic() >= deadline:
            return [], TIMEOUT_MESSAGE
        cursor.execute(query)
        if time.monotonic() >= deadline:
            return [], TIMEOUT_MESSAGE
        if cursor.description is None:
            return [], "SQL execution error: statement did not return a result set"

        records = []
        while True:
            if time.monotonic() >= deadline:
                return [], TIMEOUT_MESSAGE
            rows = cursor.fetchmany(FETCH_BATCH_SIZE)
            if time.monotonic() >= deadline:
                return [], TIMEOUT_MESSAGE
            if not rows:
                return records, ""
            records.extend(rows)
    except Exception as error:
        if interrupted_by_deadline:
            return [], TIMEOUT_MESSAGE
        return [], f"{type(error).__name__}: {error}"
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                connection.close()


def execute_queries(queries, database, timeout_seconds, workers):
    database_uri = Path(database).resolve().as_uri() + "?mode=ro"
    records = [None] * len(queries)
    errors = [""] * len(queries)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execute_query, query, database_uri, timeout_seconds): index
            for index, query in enumerate(queries)
        }
        # No overall batch timeout: waiting queries retain their full own budget.
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                records[index], errors[index] = future.result()
            except Exception as error:
                records[index] = []
                errors[index] = f"Worker error: {type(error).__name__}: {error}"
            if completed % 25 == 0 or completed == len(queries):
                print(f"Completed {completed}/{len(queries)} queries", flush=True)
    return records, errors


def main(argv=None):
    args = get_args(argv)
    source = args.predicted_sql.resolve()
    sql_dir = args.results_root.resolve() / args.eval_name
    record_dir = args.records_root.resolve() / args.eval_name
    for directory in (sql_dir, record_dir):
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {directory}; choose a fresh --eval_name.")
    for path in (source, args.database, args.reference_sql, args.reference_records):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload, queries = read_sql(source)
    _, reference_queries = read_sql(args.reference_sql)
    if len(queries) != len(reference_queries):
        raise ValueError(f"Prediction/reference counts differ: {len(queries)} versus {len(reference_queries)}")

    # Import only scoring utilities, not the training or generation pipeline.
    from utils import compute_metrics

    sql_dir.mkdir(parents=True, exist_ok=False)
    record_dir.mkdir(parents=True, exist_ok=False)
    sql_path = sql_dir / "dev.sql"
    record_path = record_dir / "dev.pkl"
    metrics_path = sql_dir / "dev_metrics.json"
    print(f"Executing {len(queries)} queries: {args.timeout_seconds:g}s per query, {args.workers} workers", flush=True)
    records, errors = execute_queries(queries, args.database, args.timeout_seconds, args.workers)

    with sql_path.open("xb") as handle:
        handle.write(payload)
    with record_path.open("xb") as handle:
        pickle.dump((records, errors), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Both record paths are supplied: compute_metrics must not execute any SQL.
    sql_em, record_em, record_f1, _ = compute_metrics(
        args.reference_sql, sql_path,
        gt_query_records=args.reference_records, model_query_records=record_path,
    )
    timeout_count = sum(error == TIMEOUT_MESSAGE for error in errors)
    other_error_count = sum(bool(error) and error != TIMEOUT_MESSAGE for error in errors)
    metrics = {
        "source_sql": str(source),
        "source_sql_sha256": hashlib.sha256(payload).hexdigest(),
        "database": str(args.database.resolve()),
        "reference_sql": str(args.reference_sql.resolve()),
        "reference_records": str(args.reference_records.resolve()),
        "example_count": len(queries),
        "timeout_seconds": args.timeout_seconds,
        "workers": args.workers,
        "sqlite_progress_steps": PROGRESS_STEPS,
        "fetch_batch_size": FETCH_BATCH_SIZE,
        "record_f1": float(record_f1),
        "record_em": float(record_em),
        "sql_em": float(sql_em),
        "error_rate": (timeout_count + other_error_count) / len(queries),
        "timeout_count": timeout_count,
        "non_timeout_execution_error_count": other_error_count,
    }
    with metrics_path.open("x", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))
    print(f"SQL copy: {sql_path}\nRecords: {record_path}\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
