"""Shared T5 tokenizer configuration for SQL data and checkpoints."""

CHECKPOINT = "google-t5/t5-small"
SQL_OPERATOR_TOKENS = ("<", "<=")


def configure_tokenizer(tokenizer):
    # Both must be ordinary tokens: <= must not decode as two operators, < =.
    # Adding existing tokens again preserves their saved IDs.
    tokenizer.add_tokens(list(SQL_OPERATOR_TOKENS), special_tokens=False)
    return tokenizer


def load_tokenizer(source=CHECKPOINT, *, add_sql_tokens=True, **kwargs):
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast.from_pretrained(source, **kwargs)
    return configure_tokenizer(tokenizer) if add_sql_tokens else tokenizer
