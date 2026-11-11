"""Text preprocessing shared by data statistics and the future T5 pipeline."""

QUESTION_PREFIX = "translate English to SQL: "


def preprocess_question(question: str) -> str:
    """Strip surrounding whitespace and add the T5 task prefix."""
    return QUESTION_PREFIX + question.strip()


def preprocess_sql(sql: str) -> str:
    """Strip surrounding whitespace without changing SQL contents."""
    return sql.strip()
