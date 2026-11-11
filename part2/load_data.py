from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import T5TokenizerFast

from preprocessing import preprocess_question, preprocess_sql

PAD_IDX = 0
LABEL_PAD_IDX = -100
DATA_DIR = Path(__file__).resolve().parent / "data"

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, tokenizer=None, max_input_length=128,
                 max_target_length=512, max_examples=None):
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Unknown split: {split}")
        if min(max_input_length, max_target_length) < 2:
            raise ValueError("Length limits must leave room for text and EOS.")
        self.tokenizer = tokenizer if tokenizer is not None else T5TokenizerFast.from_pretrained("google-t5/t5-small")
        if self.tokenizer.pad_token_id != PAD_IDX or self.tokenizer.eos_token_id != 1:
            raise ValueError("This baseline expects T5's standard PAD=0 and EOS=1.")
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_examples = max_examples
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        questions = load_lines(Path(data_folder) / f"{split}.nl")
        queries = None if split == "test" else load_lines(Path(data_folder) / f"{split}.sql")
        if queries is not None and len(questions) != len(queries):
            raise ValueError(f"{split}: question and SQL counts do not match.")
        if self.max_examples is not None:
            questions = questions[:self.max_examples]
            if queries is not None:
                queries = queries[:self.max_examples]

        # Apply shared preprocessing exactly once, before tokenization.
        questions = [preprocess_question(question) for question in questions]
        encoder_ids = tokenize_texts(tokenizer, questions)
        target_ids = None if queries is None else tokenize_texts(
            tokenizer, [preprocess_sql(sql) for sql in queries], target=True
        )
        self.length_report = {
            "examples": len(questions),
            "max_input_tokens": max(map(len, encoder_ids)),
            "truncated_inputs": sum(len(ids) > self.max_input_length for ids in encoder_ids),
            "max_target_tokens": None if target_ids is None else max(map(len, target_ids)),
            "truncated_targets": 0 if target_ids is None else sum(
                len(ids) > self.max_target_length for ids in target_ids
            ),
        }
        print(f"{split} token lengths (including EOS): {self.length_report}")
        examples = []
        for index, ids in enumerate(encoder_ids):
            ids = limit_with_eos(ids, self.max_input_length, tokenizer.eos_token_id)
            example = {
                "encoder_ids": torch.tensor(ids, dtype=torch.long),
                "encoder_mask": torch.ones(len(ids), dtype=torch.long),
                # Standard pretrained T5 starts decoding with PAD, not an extra-id.
                "initial_decoder_inputs": torch.tensor([PAD_IDX], dtype=torch.long),
            }
            if target_ids is not None:
                labels = limit_with_eos(target_ids[index], self.max_target_length, tokenizer.eos_token_id)
                example["decoder_targets"] = torch.tensor(labels, dtype=torch.long)
                example["decoder_inputs"] = torch.tensor([PAD_IDX] + labels[:-1], dtype=torch.long)
            examples.append(example)
        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def tokenize_texts(tokenizer, texts, target=False):
    token_ids = []
    for start in range(0, len(texts), 64):
        batch = texts[start:start + 64]
        inputs = {"text_target": batch} if target else {"text": batch}
        encoded = tokenizer(**inputs, add_special_tokens=True, padding=False,
                            truncation=False, return_attention_mask=False)
        token_ids.extend(encoded["input_ids"])
    return token_ids


def limit_with_eos(token_ids, maximum, eos_id):
    # Count truncation before this step, and retain EOS even when a limit is hit.
    return token_ids if len(token_ids) <= maximum else token_ids[:maximum - 1] + [eos_id]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_mask, initial_decoder_inputs = test_collate_fn(batch)
    decoder_inputs = pad_sequence([item["decoder_inputs"] for item in batch],
                                  batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([item["decoder_targets"] for item in batch],
                                   batch_first=True, padding_value=LABEL_PAD_IDX)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item["encoder_ids"] for item in batch],
                              batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item["encoder_mask"] for item in batch],
                               batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.stack([item["initial_decoder_inputs"] for item in batch])
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, tokenizer=None, data_folder=DATA_DIR,
                   max_input_length=128, max_target_length=512, max_examples=None):
    dset = T5Dataset(data_folder, split, tokenizer, max_input_length, max_target_length, max_examples)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, tokenizer=None, data_folder=DATA_DIR,
                 max_input_length=128, max_target_length=512, include_train=True,
                 include_test=False, max_train_examples=None):
    options = dict(tokenizer=tokenizer, data_folder=data_folder,
                   max_input_length=max_input_length, max_target_length=max_target_length)
    train_loader = get_dataloader(batch_size, "train", max_examples=max_train_examples, **options) if include_train else None
    dev_loader = get_dataloader(test_batch_size, "dev", **options)
    test_loader = get_dataloader(test_batch_size, "test", **options) if include_test else None
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = [line.removesuffix("\n") for line in f]
    if not lines:
        raise ValueError(f"{path}: no examples found.")
    empty = [index for index, line in enumerate(lines, 1) if not line.strip()]
    if empty:
        raise ValueError(f"{path}: empty records at lines {empty}")
    return lines

def load_prompting_data(data_folder):
    raise NotImplementedError("Prompting is outside this fine-tuning baseline.")
