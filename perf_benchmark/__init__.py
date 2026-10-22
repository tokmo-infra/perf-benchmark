import os as _os

# Configure the native batch tokenizer before Rayon first creates its process-wide
# pool. Explicit environment settings take precedence; retain fork safety by
# leaving TOKENIZERS_PARALLELISM under the tokenizer library's control.
_os.environ.setdefault("RAYON_NUM_THREADS", "4")

__version__ = "0.8.0"

__all__ = ["__version__"]
