"""Local-tokenizer prompt generation, checked against the actual chat template."""
import random
from pathlib import Path
from perf_benchmark.core.tokenization import ModelTokenizer


class VocabularyTokenSampler:
    def __init__(self, tokenizer_path=None, *, base_dir=None, template_options=None):
        if not tokenizer_path:
            raise ValueError("static benchmarking requires a local tokenizer_path matching the served model")
        self._tokenizer = ModelTokenizer(tokenizer_path, base_dir=base_dir or Path.cwd(),
                                         template_options=template_options)
        self._ordinary_tokens = self._tokenizer.ordinary

    @property
    def has_real_tokenizer(self):
        return True

    @property
    def fingerprint(self):
        return self._tokenizer.fingerprint

    def encode(self, text):
        return self._tokenizer.encode(text)

    def decode(self, token_ids):
        return self._tokenizer.decode(token_ids)

    def sample_prompt(self, target_tokens, seed=42, *, adjust_for_chat_template=False):
        if type(target_tokens) is not int or target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        rng = random.Random(seed)
        def count(text):
            return (len(self._tokenizer.encode_chat([{"role": "user", "content": text}]))
                    if adjust_for_chat_template else len(self.encode(text)))
        overhead = count("") if adjust_for_chat_template else 0
        if target_tokens <= overhead:
            raise ValueError("target shorter than chat-template overhead")
        text = self.decode(rng.choices(self._ordinary_tokens, k=target_tokens-overhead))
        for _ in range(32):
            delta = target_tokens - count(text)
            if delta == 0:
                return text
            if delta > 0:
                text += " " + self.decode(rng.choices(self._ordinary_tokens, k=delta))
            else:
                ids = self.encode(text)
                text = self.decode(ids[:max(0, len(ids)+delta)])
        raise ValueError("tokenizer could not construct the exact requested chat length")
