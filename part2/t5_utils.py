import os
import json
from importlib.metadata import version
from pathlib import Path

import torch

import transformers
from transformers import T5ForConditionalGeneration
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from preprocessing import QUESTION_PREFIX
from tokenizer_utils import SQL_OPERATOR_TOKENS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    import wandb
    config = json.loads(json.dumps(vars(args), default=str))
    return wandb.init(project="dsga1011-a4", name=args.experiment_name, config=config)

def initialize_model(args, tokenizer):
    """Initialize the pretrained baseline with every parameter trainable."""
    if not args.finetune:
        raise ValueError("This pretrained baseline requires --finetune.")
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    check_special_tokens(model)
    model.resize_token_embeddings(len(tokenizer))
    check_tokenizer_size(model, tokenizer)
    model.requires_grad_(True)
    return model.to(DEVICE)


def check_special_tokens(model):
    if (model.config.pad_token_id != 0 or model.config.decoder_start_token_id != 0
            or model.config.eos_token_id != 1):
        raise ValueError("The data pipeline expects standard T5 PAD/start=0 and EOS=1.")


def check_tokenizer_size(model, tokenizer):
    size = len(tokenizer)
    if (model.config.vocab_size != size
            or model.get_input_embeddings().weight.shape[0] != size
            or model.get_output_embeddings().weight.shape[0] != size):
        raise ValueError("Model vocabulary does not match the configured tokenizer; use a matching checkpoint.")


def run_directory(args):
    return Path(args.checkpoint_root) / "ft_experiments" / args.experiment_name


def checkpoint_path(args, best=True):
    if args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir).resolve()
    return run_directory(args) / ("best" if best else "latest")


def load_run_settings(args):
    path = checkpoint_path(args, best=not args.load_latest) / "run_settings.json"
    return json.loads(path.read_text(encoding="utf-8"))

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best, *, tokenizer, args, epoch, metrics):
    check_tokenizer_size(model, tokenizer)
    target = Path(checkpoint_dir) / ("best" if best else "latest")
    target.mkdir(parents=True, exist_ok=True)
    model.generation_config.max_new_tokens = args.generation_max_new_tokens
    model.generation_config.num_beams = args.num_beams
    model.generation_config.do_sample = False
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    settings = dict(vars(args))
    settings.update({
        "epoch": epoch,
        "dev_metrics": metrics,
        "preprocessing": {"question_prefix": QUESTION_PREFIX, "strip_question": True, "strip_sql": True},
        "tokenizer": {"size": len(tokenizer),
                      "ordinary_sql_tokens": {token: tokenizer.convert_tokens_to_ids(token)
                                              for token in SQL_OPERATOR_TOKENS}},
        "generation": {"max_new_tokens": args.generation_max_new_tokens,
                       "num_beams": args.num_beams, "do_sample": False},
        "special_tokens": {"pad": model.config.pad_token_id,
                           "decoder_start": model.config.decoder_start_token_id,
                           "eos": model.config.eos_token_id,
                           "label_padding": -100},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "tokenizers": version("tokenizers")},
    })
    (target / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n", encoding="utf-8"
    )

def load_model_from_checkpoint(args, best, tokenizer):
    path = checkpoint_path(args, best)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = T5ForConditionalGeneration.from_pretrained(path, local_files_only=True)
    check_special_tokens(model)
    check_tokenizer_size(model, tokenizer)
    return model.to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_type}")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs
    args.steps_per_epoch = epoch_length
    args.planned_training_steps = num_training_steps
    args.warmup_steps = num_warmup_steps

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

