import json
import os
import socket
import subprocess
import sys

import pytest

from perf_benchmark.core.tokenization import ModelTokenizer


@pytest.mark.parametrize("workers,parallel,expected", [
    (None, None, 4),
    ("2", None, 2),
    (None, "false", 0),
])
def test_native_batch_thread_count_and_environment_override(workers, parallel, expected):
    # A fresh process is essential: Rayon initializes one pool for its lifetime.
    env = dict(os.environ)
    for name, value in (("RAYON_NUM_THREADS", workers), ("TOKENIZERS_PARALLELISM", parallel)):
        env.pop(name, None)
        if value is not None:
            env[name] = value
    result = subprocess.run([sys.executable, "-c", """
import json
import perf_benchmark
import psutil
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "one": 1, "two": 2}, unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
process = psutil.Process()
before = process.num_threads()
texts = ["one two", "", "two one one"] * 256
encoded = tokenizer.encode_batch(texts, add_special_tokens=False)
assert [value.ids for value in encoded] == [[1, 2], [], [2, 1, 1]] * 256
assert tokenizer.decode_batch([value.ids for value in encoded]) == texts
print(json.dumps({"threads": process.num_threads() - before}))
"""], env=env, capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout)["threads"] == expected


def test_local_text_counting_without_template(tokenizer_path, monkeypatch):
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        return original(path, **kwargs)
    def blocked(*args, **kwargs):
        raise AssertionError("local tokenization attempted network access")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    tokenizer = ModelTokenizer(tokenizer_path.name, base_dir=tokenizer_path.parent)
    assert calls == [{"local_files_only": True, "trust_remote_code": False, "use_fast": True}]
    texts = ["one two three", "", "four five"]
    assert tokenizer.count_batch(texts) == [3, 0, 2]
    assert tokenizer.count_batch(texts) == [len(tokenizer.encode(text)) for text in texts]
    assert tokenizer.count_batch([]) == []
    assert isinstance(tokenizer.decode(tokenizer.encode("one")), str)
    assert tokenizer.identity["template"] is None
    with pytest.raises(ValueError, match="chat template"):
        tokenizer.encode_chat([{"role": "user", "content": "one"}])


def test_invalid_tokenizer_configuration(tokenizer_path):
    # Missing remote IDs and plain files must both be rejected, never downloaded.
    for path in (None, "remote/model-name", tokenizer_path / "tokenizer.json"):
        with pytest.raises(ValueError, match="tokenizer path"):
            ModelTokenizer(path, base_dir=tokenizer_path.parent)
    for options in ({"enable_thinking": "false"}, {"unknown": True}):
        with pytest.raises(ValueError, match="template_options"):
            ModelTokenizer(tokenizer_path, template_options=options)


def test_chat_encoding_fingerprint_options_and_no_body_mutation(tokenizer_path):
    (tokenizer_path / "chat_template.jinja").write_text(
        "{% if tools %}{{ tools | tojson }} {% endif %}"
        "{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    tokenizer = ModelTokenizer(tokenizer_path)
    messages = [{"role": "system", "content": "one two"}, {"role": "user", "content": "three"}]
    before = json.dumps(messages)
    assert len(tokenizer.encode_chat(messages)) == 6
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    assert len(tokenizer.encode_chat(messages, tools=tools)) > 6
    assert json.dumps(messages) == before
    assert tokenizer.fingerprint == ModelTokenizer(tokenizer_path).fingerprint
    changed = ModelTokenizer(tokenizer_path, template_options={"add_generation_prompt": False})
    assert changed.fingerprint != tokenizer.fingerprint
    assert len(changed.encode_chat(messages)) == 5
    (tokenizer_path / "chat_template.jinja").write_text("another {{ messages[0].content }}")
    assert ModelTokenizer(tokenizer_path).fingerprint != tokenizer.fingerprint


def test_template_error_does_not_expose_private_content(tokenizer_path):
    (tokenizer_path / "chat_template.jinja").write_text("{{ raise_exception(messages[0].content) }}")
    tokenizer = ModelTokenizer(tokenizer_path)
    with pytest.raises(ValueError, match="template rejected") as error:
        tokenizer.encode_chat([{"role": "user", "content": "private-prompt-marker"}])
    assert "private-prompt-marker" not in str(error.value)
    with pytest.raises(ValueError, match="template rejected") as error:
        tokenizer.count_chat_batch([[{"role": "user", "content": "private-prompt-marker"}]])
    assert "private-prompt-marker" not in str(error.value)


def test_native_batch_equivalence(tokenizer_path):
    (tokenizer_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role }} {{ message.content }} {% endfor %}"
        "{% if add_generation_prompt %}assistant{% endif %}"
    )
    tokenizer = ModelTokenizer(tokenizer_path)
    sequences = [[], tokenizer.encode("one two three"), tokenizer._tokenizer.all_special_ids]
    assert tokenizer.decode_batch(sequences) == [tokenizer.decode(ids) for ids in sequences]
    assert tokenizer.decode_batch([]) == []
    conversations = [[{"role": "user", "content": text}] for text in ("", "one two", "中文 🙂\n")]
    assert tokenizer.count_chat_batch(conversations) == [len(tokenizer.encode_chat(messages)) for messages in conversations]
    assert tokenizer.count_chat_batch([]) == []


def test_chat_encoding_requests_token_ids_even_when_backend_defaults_to_dict(tokenizer_path, monkeypatch):
    (tokenizer_path / "chat_template.jinja").write_text("{{ messages[0].content }} assistant")
    tokenizer = ModelTokenizer(tokenizer_path)
    original = tokenizer._tokenizer.apply_chat_template
    def backend(*args, **kwargs):
        kwargs.setdefault("return_dict", True)
        return original(*args, **kwargs)
    monkeypatch.setattr(tokenizer._tokenizer, "apply_chat_template", backend)
    messages = [{"role": "user", "content": "one two three"}]
    assert tokenizer.encode_chat(messages) == [0, 0, 0, 0]
    assert tokenizer.count_chat_batch([messages]) == [4]
