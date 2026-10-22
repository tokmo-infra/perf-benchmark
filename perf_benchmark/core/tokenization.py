"""Local model tokenization shared by preparation and offline workload tools.

No pipeline context, dataset recipe, request protocol, or inference backend is needed.
Callers perform loading and encoding on their preparation worker thread.
"""

from functools import cached_property
import hashlib
import json
import logging
from pathlib import Path


class ModelTokenizer:
    def __init__(self, path, *, base_dir=Path("."), template_options=None):
        if not path:
            raise ValueError("tokenizer path is required")
        path = Path(path).expanduser()
        self.path = (path if path.is_absolute() else Path(base_dir) / path).resolve()
        if not self.path.is_dir():
            raise ValueError("tokenizer path must be an existing local directory")
        self.options = dict(template_options or {})
        if set(self.options) - {"add_generation_prompt", "enable_thinking"} or not all(isinstance(value, bool) for value in self.options.values()):
            raise ValueError("template_options supports only boolean add_generation_prompt/enable_thinking")
        self.options.setdefault("add_generation_prompt", True)

        # Only the inference-backend advisory is irrelevant for tokenizer-only use.
        logger = logging.getLogger("transformers")
        def tokenizer_only(record):
            return not record.getMessage().startswith("None of PyTorch, TensorFlow >= 2.0, or Flax have been found.")
        logger.addFilter(tokenizer_only)
        try:
            from transformers import AutoTokenizer
        finally:
            logger.removeFilter(tokenizer_only)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if not self._tokenizer.is_fast:
            raise ValueError("a fast Transformers tokenizer is required")

    def encode(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False, verbose=False)

    def count_batch(self, texts):
        if not texts:
            return []
        encoded = self._tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False, verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def decode(self, tokens):
        return self._tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def decode_batch(self, sequences):
        # Fast tokenizers perform this bounded batch in Rust, including parallel work.
        # No cleanup: preserve the same bytes as decode().
        return self._tokenizer.backend_tokenizer.decode_batch(sequences, skip_special_tokens=False)

    def count_chat_batch(self, conversations):
        if not conversations:
            return []
        template = self.template
        try:
            encoded = self._tokenizer.apply_chat_template(
                conversations, chat_template=template, tokenize=True, return_dict=False,
                tokenizer_kwargs={"verbose": False}, **self.options,
            )
        except Exception:
            raise ValueError("local chat template rejected the snapshot") from None
        return [len(ids) for ids in encoded]

    @cached_property
    def template(self):
        if not self._tokenizer.chat_template:
            raise ValueError("local tokenizer must provide a chat template")
        return self._tokenizer.get_chat_template()

    def encode_chat(self, messages, *, tools=None):
        template = self.template
        try:
            return self._tokenizer.apply_chat_template(
                messages, tools=tools, chat_template=template, tokenize=True, return_dict=False,
                tokenizer_kwargs={"verbose": False}, **self.options,
            )
        except Exception:
            # Jinja raise_exception() may contain private message text.
            raise ValueError("local chat template rejected the snapshot") from None

    @cached_property
    def ordinary(self):
        values = sorted(set(self._tokenizer.get_vocab().values()) - set(self._tokenizer.all_special_ids))
        if not values:
            raise ValueError("tokenizer has no ordinary tokens")
        return values

    @cached_property
    def identity(self):
        files = {}
        for item in sorted(self.path.rglob("*")):
            if item.is_file() and item.suffix in {".json", ".jinja", ".model", ".txt"}:
                with item.open("rb") as stream:
                    files[str(item.relative_to(self.path))] = hashlib.file_digest(stream, "sha256").hexdigest()
        template = hashlib.sha256(self.template.encode()).hexdigest() if self._tokenizer.chat_template else None
        return {"files": files, "template": template, "template_options": self.options}

    @cached_property
    def fingerprint(self):
        encoded = json.dumps(self.identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()
