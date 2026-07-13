"""Tokenizer loading + chat-prompt rendering."""

from transformers import AutoTokenizer


def load_tokenizer(model_name: str, cache_dir: str):
    return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


def render_chat(tokenizer, messages: list[dict]) -> str:
    """Wrap a list of {role, content} messages in the model's chat format."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
