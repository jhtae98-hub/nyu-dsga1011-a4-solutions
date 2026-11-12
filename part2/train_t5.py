import os
import argparse
import json
import math
import pickle
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from t5_utils import (initialize_model, initialize_optimizer_and_scheduler, save_model,
                      load_model_from_checkpoint, setup_wandb, run_directory,
                      checkpoint_path, load_run_settings)
from tokenizer_utils import load_tokenizer
from load_data import load_t5_data, LABEL_PAD_IDX
from utils import compute_metrics, set_random_seeds
from reevaluate_sql import execute_queries

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0
SCRIPT_DIR = Path(__file__).resolve().parent

def get_args(argv=None):
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Fine-tune pretrained T5; required for training")
    parser.add_argument('--model_name', default='google-t5/t5-small')
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=5,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=3,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_input_length', type=int, default=128)
    parser.add_argument('--max_target_length', type=int, default=512)
    parser.add_argument('--generation_max_new_tokens', type=int, default=512)
    parser.add_argument('--num_beams', type=int, default=1, help='1 gives greedy decoding; sampling is disabled')
    # None distinguishes an omitted evaluation option from an explicit override.
    parser.add_argument('--sql_timeout_seconds', type=float, default=None,
                        help='Per-query execution deadline; default 10 seconds, or saved value in evaluation-only mode')
    parser.add_argument('--sql_workers', type=int, default=None,
                        help='SQL execution workers; default 2, or saved value in evaluation-only mode')
    parser.add_argument('--max_train_examples', type=int, default=None, help='Optional small training subset for smoke tests')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=Path, default=SCRIPT_DIR / 'data')
    parser.add_argument('--checkpoint_root', type=Path, default=SCRIPT_DIR / 'checkpoints')
    parser.add_argument('--results_root', type=Path, default=SCRIPT_DIR / 'results')
    parser.add_argument('--records_root', type=Path, default=SCRIPT_DIR / 'records')
    parser.add_argument('--eval_only', action='store_true', help='Load a checkpoint and evaluate without training')
    parser.add_argument('--eval_num_beams', type=int, default=None,
                        help='Evaluation-only beam override applied after restoring checkpoint settings')
    parser.add_argument('--eval_name', default=None, help='Evaluation output folder name; does not select a checkpoint')
    parser.add_argument('--checkpoint_dir', type=Path, default=None, help='Explicit best/ or latest/ checkpoint directory')
    parser.add_argument('--load_latest', action='store_true', help='Select latest instead of best in evaluation-only mode')
    parser.add_argument('--predict_test', action='store_true', help='Opt in to test loading, generation, and record export')
    parser.add_argument('--export_name', default=None, help='Test export stem; experiment gives assignment filenames')

    args = parser.parse_args(argv)
    positive = ('batch_size', 'test_batch_size', 'max_n_epochs', 'patience_epochs',
                'generation_max_new_tokens', 'num_beams')
    if any(getattr(args, key) < 1 for key in positive):
        parser.error('Batch sizes, epochs, patience, generation length, and beams must be positive.')
    if min(args.max_input_length, args.max_target_length) < 2:
        parser.error('Input and target limits must be at least 2.')
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.num_warmup_epochs < 0:
        parser.error('Learning rate must be positive; weight decay and warmup must be nonnegative.')
    if args.max_train_examples is not None and args.max_train_examples < 1:
        parser.error('--max_train_examples must be positive.')
    try:
        validate_sql_settings(args.sql_timeout_seconds, args.sql_workers)
        if not args.eval_only:
            resolve_sql_settings(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.eval_only and not args.finetune:
        parser.error('Use --finetune for this pretrained baseline, or --eval_only to load a checkpoint.')
    if not args.eval_only and (args.checkpoint_dir is not None or args.load_latest):
        parser.error('--checkpoint_dir and --load_latest require --eval_only.')
    if not args.eval_only and (args.eval_num_beams is not None or args.eval_name is not None):
        parser.error('--eval_num_beams and --eval_name require --eval_only.')
    if args.eval_num_beams is not None:
        if args.eval_num_beams < 1:
            parser.error('--eval_num_beams must be positive.')
        if args.eval_name is None:
            parser.error('--eval_num_beams requires a separate --eval_name to preserve baseline outputs.')
    if args.export_name is not None and not args.predict_test:
        parser.error('--export_name requires --predict_test.')
    for name in (args.experiment_name, args.export_name, args.eval_name):
        if name is not None and (not name or name in ('.', '..') or '/' in name or '\\' in name):
            parser.error('Experiment, evaluation, and export names must be single nonempty path components.')
    return args


def validate_sql_settings(timeout_seconds, workers):
    if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('--sql_timeout_seconds must be finite and positive.')
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError('--sql_workers must be a positive integer.')


def resolve_sql_settings(args, saved=None):
    saved = saved or {}
    for key, default in (('sql_timeout_seconds', 10.0), ('sql_workers', 2)):
        if getattr(args, key) is None:
            value = saved.get(key)
            setattr(args, key, default if value is None else value)
    validate_sql_settings(args.sql_timeout_seconds, args.sql_workers)


def restore_checkpoint_settings(args):
    saved = load_run_settings(args)
    for key in ('experiment_name', 'finetune', 'model_name', 'seed', 'max_input_length',
                'max_target_length', 'generation_max_new_tokens', 'num_beams'):
        setattr(args, key, saved[key])
    # Explicit CLI values win; older checkpoints fall back to 10 seconds / 2 workers.
    resolve_sql_settings(args, saved)
    if args.eval_num_beams is not None:
        if args.eval_name == args.experiment_name:
            raise ValueError('--eval_name must differ from the checkpoint experiment name when overriding beams.')
        args.num_beams = args.eval_num_beams
    return checkpoint_path(args, best=not args.load_latest)


def evaluation_settings(args):
    return dict(checkpoint=str(checkpoint_path(args, best=not args.load_latest).resolve()),
                eval_name=args.eval_name if args.eval_name is not None else args.experiment_name,
                num_beams=args.num_beams,
                generation_max_new_tokens=args.generation_max_new_tokens, do_sample=False,
                sql_timeout_seconds=args.sql_timeout_seconds, sql_workers=args.sql_workers)


def output_paths(args, split):
    name = args.eval_name if args.eval_name is not None else args.experiment_name
    sql_dir = Path(args.results_root) / name
    record_dir = Path(args.records_root) / name
    sql_name, record_name = f'{split}.sql', f'{split}.pkl'
    if split == 'test' and args.export_name is not None:
        sql_dir, record_dir = Path(args.results_root), Path(args.records_root)
        sql_name = f't5_ft_{args.export_name}_test.sql'
        record_name = f't5_ft_{args.export_name}_test.pkl'
    sql_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    return sql_dir / sql_name, record_dir / record_name


def save_sql_predictions(args, predictions, sql_path, record_path):
    database = args.data_dir / 'flight_database.db'
    if not database.is_file():
        raise FileNotFoundError(database)
    if any('\n' in query or '\r' in query for query in predictions):
        raise ValueError('Each exported SQL query must occupy one line.')
    sql_path, record_path = Path(sql_path), Path(record_path)
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with sql_path.open('w', encoding='utf-8') as handle:
        for query in predictions:
            handle.write(f'{query}\n')
    records, errors = execute_queries(predictions, database, args.sql_timeout_seconds, args.sql_workers)
    if len(records) != len(predictions) or len(errors) != len(predictions):
        raise ValueError('SQL execution lost example alignment.')
    with record_path.open('wb') as handle:
        pickle.dump((records, errors), handle)
    return records, errors


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    checkpoint_dir = run_directory(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    gt_sql_path = args.data_dir / 'dev.sql'
    gt_record_path = args.records_root / 'ground_truth_dev.pkl'
    model_sql_path, model_record_path = output_paths(args, 'dev')
    history = []
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch + 1}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch + 1}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch + 1}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            import wandb
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        metrics = dict(loss=eval_loss, record_f1=record_f1, record_em=record_em,
                       sql_em=sql_em, error_rate=error_rate,
                       sql_timeout_seconds=args.sql_timeout_seconds, sql_workers=args.sql_workers)
        history.append(dict(epoch=epoch + 1, train_loss=tr_loss, **metrics))
        (checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
        save_options = dict(tokenizer=dev_loader.dataset.tokenizer, args=args, epoch=epoch + 1, metrics=metrics)
        save_model(checkpoint_dir, model, best=False, **save_options)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True, **save_options)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_IDX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        non_pad = decoder_targets != LABEL_PAD_IDX
        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=non_pad,
            use_cache=False,
        )['logits']

        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=LABEL_PAD_IDX, reduction='sum')
    total_loss, total_tokens, limit_hits = 0.0, 0, 0
    predictions = []
    tokenizer = dev_loader.dataset.tokenizer
    with torch.no_grad():
        for encoder_ids, encoder_mask, decoder_inputs, targets, initial in tqdm(dev_loader):
            encoder_ids, encoder_mask = encoder_ids.to(DEVICE), encoder_mask.to(DEVICE)
            decoder_inputs, targets = decoder_inputs.to(DEVICE), targets.to(DEVICE)
            non_pad = targets != LABEL_PAD_IDX
            logits = model(input_ids=encoder_ids, attention_mask=encoder_mask,
                           decoder_input_ids=decoder_inputs, decoder_attention_mask=non_pad,
                           use_cache=False).logits
            total_loss += criterion(logits[non_pad], targets[non_pad]).item()
            total_tokens += non_pad.sum().item()
            decoded, hits = generate_sql(args, model, tokenizer, encoder_ids, encoder_mask, initial.to(DEVICE))
            predictions.extend(decoded)
            limit_hits += hits
    if len(predictions) != len(dev_loader.dataset):
        raise ValueError('Development generation lost example alignment.')
    print(f'Dev generations without EOS at the length limit: {limit_hits}/{len(predictions)}')
    save_sql_predictions(args, predictions, model_sql_path, model_record_path)
    # The supplied utility returns SQL EM, Record EM, Record F1, then errors.
    sql_em, record_em, record_f1, errors = compute_metrics(
        gt_sql_pth, model_sql_path, gt_query_records=gt_record_path, model_query_records=model_record_path
    )
    error_rate = sum(bool(error) for error in errors) / len(predictions)
    return total_loss / total_tokens, float(record_f1), float(record_em), float(sql_em), error_rate


def generate_sql(args, model, tokenizer, encoder_ids, encoder_mask, initial_decoder_inputs):
    generated = model.generate(
        input_ids=encoder_ids,
        attention_mask=encoder_mask,
        decoder_input_ids=initial_decoder_inputs,
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=args.generation_max_new_tokens,
        num_beams=args.num_beams,
        do_sample=False,
    )
    limit_hits = (~generated.eq(model.config.eos_token_id).any(dim=1)).sum().item()
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [query.strip() for query in decoded], limit_hits
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    predictions, limit_hits = [], 0
    with torch.no_grad():
        for encoder_ids, encoder_mask, initial in tqdm(test_loader):
            decoded, hits = generate_sql(args, model, test_loader.dataset.tokenizer,
                                         encoder_ids.to(DEVICE), encoder_mask.to(DEVICE), initial.to(DEVICE))
            predictions.extend(decoded)
            limit_hits += hits
    if len(predictions) != len(test_loader.dataset):
        raise ValueError('Test generation lost example alignment.')
    print(f'Test generations without EOS at the length limit: {limit_hits}/{len(predictions)}')
    save_sql_predictions(args, predictions, model_sql_path, model_record_path)
    print(f'Test queries: {model_sql_path}\nTest records: {model_record_path}')

def main():
    # Get key arguments
    args = get_args()
    if args.eval_only:
        # Restore saved defaults first, then apply only the explicit evaluation override.
        tokenizer_source = restore_checkpoint_settings(args)
    else:
        if run_directory(args).exists():
            raise FileExistsError(f'{run_directory(args)} already exists; choose a new --experiment_name.')
        tokenizer_source = args.model_name
    set_random_seeds(args.seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    tokenizer = load_tokenizer(tokenizer_source, local_files_only=args.eval_only)
    wandb_run = setup_wandb(args) if args.use_wandb else None

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size, tokenizer=tokenizer, data_folder=args.data_dir,
        max_input_length=args.max_input_length, max_target_length=args.max_target_length,
        include_train=not args.eval_only, include_test=args.predict_test,
        max_train_examples=args.max_train_examples,
    )
    args.data_statistics = {
        split: loader.dataset.length_report for split, loader in
        (('train', train_loader), ('dev', dev_loader), ('test', test_loader)) if loader is not None
    }

    # Train 
    if not args.eval_only:
        model = initialize_model(args, tokenizer)
        print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        # Release the training model and optimizer before loading best weights.
        del model, optimizer, scheduler

    # Evaluate
    model = load_model_from_checkpoint(args, best=not args.load_latest, tokenizer=tokenizer)
    model.eval()
    
    # Dev set
    gt_sql_path = args.data_dir / 'dev.sql'
    gt_record_path = args.records_root / 'ground_truth_dev.pkl'
    model_sql_path, model_record_path = output_paths(args, 'dev')
    resolved_settings = evaluation_settings(args)
    print('Resolved evaluation settings: ' + json.dumps(resolved_settings, sort_keys=True), flush=True)
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")
    metrics = dict(loss=dev_loss, record_f1=dev_record_f1, record_em=dev_record_em,
                   sql_em=dev_sql_em, error_rate=dev_error_rate,
                   **resolved_settings)
    (model_sql_path.parent / 'dev_metrics.json').write_text(
        json.dumps(metrics, indent=2) + '\n', encoding='utf-8'
    )

    # Test set
    if args.predict_test:
        model_sql_path, model_record_path = output_paths(args, 'test')
        test_inference(args, model, test_loader, model_sql_path, model_record_path)
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
