from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Union

import regex  # pip install regex

# GPT-2 style tokenizer regex (same as in your Rust code)
GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Equivalent of static Lazy<FRegex>
SLOW_RE = regex.compile(GPT2_TOKENIZER_REGEX)


def decode_utf8_ignore(b: bytes) -> str:
    """
    Rough equivalent of the Rust decode_utf8_ignore:
    - If valid UTF-8, return normal string
    - If invalid, replace bad sequences with U+FFFD (�) instead of failing
    """
    # Python's decode(errors="replace") already does the "lossy" behavior.
    return b.decode("utf-8", errors="replace")


class Splitter:
    """
    Python version of the Rust Splitter.

    - special_token is treated literally, escaped into a regex.
    - split() accepts bytes or str and returns {token: count}.
    - seek_and_split() reads a byte range [start, end) from file and does the same.
    """

    def __init__(self, special_token: str):
        # Equivalent to Regex::new(&escape(&special_token))
        self.split_re = re.compile(re.escape(special_token))

    def split(self, chunk: Union[bytes, str]) -> Dict[str, int]:
        """
        Split a bytes or str object into token-like substrings
        and return a dict[token] = count.
        """
        if isinstance(chunk, bytes):
            text = decode_utf8_ignore(chunk)
        elif isinstance(chunk, str):
            text = chunk
        else:
            raise TypeError("Expected bytes or str")

        return self._split_counts_internal(text)

    def seek_and_split(self, filepath: str, start: int, end: int) -> Dict[str, int]:
        """
        Read a byte range [start, end) from a file and split.
        """
        if end < start:
            raise ValueError("end < start")

        with open(filepath, "rb") as f:
            f.seek(start)
            buf = f.read(end - start)

        text = decode_utf8_ignore(buf)
        return self._split_counts_internal(text)

    def _split_counts_internal(self, text: str) -> Dict[str, int]:
        """
        Internal logic: split on special_token, then apply GPT-2 regex on each segment.
        """
        counts = Counter()

        # Split on special token; the special token itself is discarded.
        for segment in self.split_re.split(text):
            if not segment:
                continue
            # Apply GPT-2 regex to this segment
            for m in SLOW_RE.finditer(segment):
                tok = m.group(0)
                counts[tok] += 1

        return dict(counts)
