import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import re
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

_WORD_SPANS = re.compile(r"<[^>]*>|\b\w+(?:['\u2019]\w+)*\b")
_PROTECTED_WORDS = {
    "cannot", "neither", "never", "no", "nobody", "none",
    "nor", "not", "nothing", "nowhere", "rarely", "without",
}
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    def replace_word(match):
        word = match.group(0)
        # Skip tags, contractions, short/non-ASCII words, and protected cues.
        if re.fullmatch(r"[A-Za-z]{5,}", word) is None or word.lower() in _PROTECTED_WORDS:
            return word
        if random.random() >= 0.50:
            return word
        position = random.randrange(1, len(word) - 1)
        letter = word[position]
        row = next(row for row in _KEYBOARD_ROWS if letter.lower() in row)
        key_index = row.index(letter.lower())
        neighbors = []
        if key_index > 0:
            neighbors.append(row[key_index - 1])
        if key_index < len(row) - 1:
            neighbors.append(row[key_index + 1])
        replacement = random.choice(neighbors)
        if letter.isupper():
            replacement = replacement.upper()
        return word[:position] + replacement + word[position + 1:]
    # Substitution preserves unmatched characters and processes matches left to right.
    example = example.copy()
    example["text"] = _WORD_SPANS.sub(replace_word, example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
