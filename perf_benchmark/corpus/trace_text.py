"""Length-only text for production traces without known prefix blocks."""


def plain_text(tokenizer, target, rng):
    # Length-only sources contain no prefix evidence. Keep their text request-specific.
    ids = tokenizer.ordinary
    sample = tokenizer.decode(rng.choices(ids, k=target))
    for _ in range(64):
        actual = len(tokenizer.encode(sample))
        if actual == target:
            return sample
        delta = target - actual
        if delta > 0:
            sample += tokenizer.decode(rng.choices(ids, k=delta))
        else:
            encoded = tokenizer.encode(sample)
            sample = tokenizer.decode(encoded[:target])
    raise ValueError(f"cannot construct exact {target}-token text for this tokenizer")
