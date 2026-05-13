import regex as re
from pathlib import Path
import math
import os
from cs336_basics.tokenization.pretokenization_example import find_chunk_boundaries
from typing import Dict, List, Tuple, Optional
from argparse import ArgumentParser
import numpy as np
import json
import time
import logging
import pickle


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:
    """
    GPT-2 style byte-level BPE tokenizer.

    Assumptions:
    - `vocab` is a dict[int, bytes] mapping token ids -> token byte sequences.
    - Single-byte tokens exist in the vocab and are mapped by their byte value
      (e.g. bytes([b]) has some id in the vocab).
    - `merges` is a list of (left_token_bytes, right_token_bytes) pairs in
      merge-order.
    """

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab: Dict[int, bytes] = vocab
        # Reverse map: token bytes -> id
        self.bytes2id: Dict[bytes, int] = {v: k for k, v in vocab.items()}

        # --------- special tokens ---------
        self.special_tokens: List[str] = special_tokens or []

        # Map special-token *strings* to ids (we assume they exist in vocab
        # encoded as UTF-8).
        self.special_token_to_id: Dict[str, int] = {}
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b not in self.bytes2id:
                raise ValueError(f"Special token {st!r} not found in vocab")
            self.special_token_to_id[st] = self.bytes2id[b]

        # Build a regex that:
        # - matches the *longest* special tokens first
        # - preserves them as separate segments via a capturing group
        if self.special_tokens:
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(re.escape(t) for t in specials_sorted) + ")"
            self.special_split_re = re.compile(pattern)
        else:
            self.special_split_re = None

        # --------- BPE merges in id-space ---------
        # For each merge (left_bytes, right_bytes), we build:
        # - ids for left and right tokens
        # - id for the merged token (left_bytes + right_bytes)
        # and store:
        #   bpe_ranks[(left_id, right_id)] = rank
        #   merge_result[(left_id, right_id)] = merged_id
        self.bpe_ranks: Dict[Tuple[int, int], int] = {}
        self.merge_result: Dict[Tuple[int, int], int] = {}

        for rank, (l_bytes, r_bytes) in enumerate(merges):
            if l_bytes not in self.bytes2id or r_bytes not in self.bytes2id:
                # If some merge refers to tokens not in vocab, skip it.
                # (This shouldn't happen for a well-formed model.)
                continue

            left_id = self.bytes2id[l_bytes]
            right_id = self.bytes2id[r_bytes]
            merged_bytes = l_bytes + r_bytes
            if merged_bytes not in self.bytes2id:
                # Again, shouldn't happen in a proper GPT-2-style model.
                continue
            merged_id = self.bytes2id[merged_bytes]

            pair = (left_id, right_id)
            self.bpe_ranks[pair] = rank
            self.merge_result[pair] = merged_id


    @classmethod
    def from_files(
            cls,
            vocab_path: str,
            merges_path: str,
            special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        """
        Load tokenizer from:
          - vocab_path: JSON dict {token_str: id} or lines "id<TAB>repr"
          - merges_path: lines "left right" or "b'..'\tb'..'"
        Adapt here to your actual formats. This version assumes:
          * vocab.json: {"<|endoftext|>": 50256, " hello": 313, ...}
          * merges.txt: one merge per line like "h e" (GPT-2 style).
        """

        vocab_file = Path(vocab_path)
        merges_file = Path(merges_path)

        # ---- Load vocab ----
        # If it's JSON: {"token": idx, ...}
        with vocab_file.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "{":
                # JSON format
                raw = json.load(f)  # token_str -> id
                vocab: Dict[int, bytes] = {
                    int(idx): tok.encode("utf-8") for tok, idx in raw.items()
                }
            else:
                # Fallback: "idx<TAB>repr" with repr being a Python bytes literal
                import ast

                vocab = {}
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    idx_str, byte_repr = line.split("\t", 1)
                    idx = int(idx_str)
                    token_bytes = ast.literal_eval(byte_repr)
                    if isinstance(token_bytes, bytearray):
                        token_bytes = bytes(token_bytes)
                    assert isinstance(token_bytes, bytes)
                    vocab[idx] = token_bytes

        # ---- Load merges ----
        merges: List[Tuple[bytes, bytes]] = []
        with merges_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Try GPT-2 style "a b"
                parts = line.split()
                if len(parts) == 2 and not parts[0].startswith("b'"):
                    left, right = parts
                    merges.append((left.encode("utf-8"), right.encode("utf-8")))
                    continue

                # Fallback: "b'..'\tb'..'" format
                import ast

                left_repr, right_repr = line.split("\t", 1)
                left = ast.literal_eval(left_repr)
                right = ast.literal_eval(right_repr)
                if isinstance(left, bytearray):
                    left = bytes(left)
                if isinstance(right, bytearray):
                    right = bytes(right)
                assert isinstance(left, bytes) and isinstance(right, bytes)
                merges.append((left, right))


        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # ---------------------------------------------------------------------
    # Core BPE
    # ---------------------------------------------------------------------

    @staticmethod
    def _get_pairs(symbols: List[int]) -> set[Tuple[int, int]]:
        pairs = set()
        prev = symbols[0]
        for s in symbols[1:]:
            pairs.add((prev, s))
            prev = s
        return pairs

    def _bpe_ids(self, token_bytes: bytes) -> List[int]:
        """
        Apply BPE to a single piece of text given as raw bytes.

        Returns a list of token ids (int).
        """
        if not token_bytes:
            return []

        # Start with single-byte tokens.
        try:
            symbols = [self.bytes2id[bytes([b])] for b in token_bytes]
        except KeyError as e:
            raise KeyError(f"Single-byte token {bytes([e.args[0]])!r} "
                           f"not found in vocab") from e

        if len(symbols) == 1:
            return symbols

        pairs = self._get_pairs(symbols)

        while True:
            # Find best-ranked pair present in `pairs`.
            min_rank = None
            best_pair = None
            for p in pairs:
                rank = self.bpe_ranks.get(p)
                if rank is None:
                    continue
                if min_rank is None or rank < min_rank:
                    min_rank = rank
                    best_pair = p

            if best_pair is None:
                break  # no more applicable merges

            left, right = best_pair
            merged_id = self.merge_result[best_pair]

            # Merge all occurrences of best_pair.
            new_symbols: List[int] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    new_symbols.append(merged_id)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols
            if len(symbols) == 1:
                break
            pairs = self._get_pairs(symbols)

        return symbols

    # ---------------------------------------------------------------------
    # Public API: encode / decode
    # ---------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Encode a Unicode string into GPT-2 style token ids.

        - Respects `special_tokens` as atomic segments.
        - Uses PAT for GPT-2-style pretokenization of non-special text.
        """
        out: List[int] = []

        if self.special_split_re is not None:
            segments = self.special_split_re.split(text)
        else:
            segments = [text]

        for segment in segments:
            if segment == "" or segment is None:
                continue

            # Special token: emit directly.
            if segment in self.special_token_to_id:
                out.append(self.special_token_to_id[segment])
                continue

            # Normal text: apply GPT-2 pre-tokenization regex and then BPE.
            for m in PAT.finditer(segment):
                piece = m.group(0)
                piece_bytes = piece.encode("utf-8")
                out.extend(self._bpe_ids(piece_bytes))

        return out

    def decode(self, token_ids: List[int] | int) -> str:
        """
        Decode a list of token ids (or a single id) back to text.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        b = b"".join(self.vocab[i] for i in token_ids)
        return b.decode("utf-8", errors="replace")

    # ---------------------------------------------------------------------
    # Convenience / streaming
    # ---------------------------------------------------------------------

    def encode_iterable(self, iterable) -> List[int]:
        """
        Encode an iterable of strings and yield token ids sequentially.
        """
        from tqdm.auto import tqdm

        for text in tqdm(iterable):
            for tid in self.encode(text):
                yield tid

    def encode_file(self, input_path: str | os.PathLike) -> list[int]:
        """
        Stream-encode a large file by chunks.

        We use `find_chunk_boundaries` from `pretokenization_example.py`,
        which expects a single *bytes* object as the split special token.
        We therefore take the first special token (if any) and use that.
        """
        file_path = Path(input_path)
        chunk_size_in_bytes = 1024 * 1024 * 10  # 10 MB
        n_chunks = math.ceil(file_path.stat().st_size / chunk_size_in_bytes)

        # pretokenization_example.find_chunk_boundaries expects ONE bytes token,
        # not a list.
        if self.special_tokens:
            split_special_token = self.special_tokens[0].encode("utf-8")
        else:
            # If you have no special tokens, you can pass b"" (or change
            # find_chunk_boundaries to handle None); b"" is at least bytes.
            split_special_token = b""

        tokens: list[int] = []
        with file_path.open("rb") as f:
            boundaries = find_chunk_boundaries(f, n_chunks, split_special_token)
            f.seek(0)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start)
                # Decode chunk to text for reuse of encode()
                text = chunk.decode("utf-8", errors="replace")
                tokens.extend(self.encode(text))

        return tokens



def parse_args():
    p = ArgumentParser()
    p.add_argument("--vocab-path", default="/home/katinska/Stanford_CS336/cs336_basics/bpe_model_tiny_stories/bpe-vocab.txt")
    p.add_argument("--merges-path", default="/home/katinska/Stanford_CS336/cs336_basics/bpe_model_tiny_stories/bpe-merges.txt")
    p.add_argument("--data-path", default="/home/katinska/Stanford_CS336/data")
    p.add_argument("--tokenized-data-path", default="/home/katinska/Stanford_CS336/data/tokenized_data_tiny_stories")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tok = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )
    input_dir = Path(args.data_path)

    compression_ratios = {}
    for f in input_dir.glob("*-valid.txt"):
        # sample 100 docs from each dataset
        ratios = []
        for doc in re.split(tok.special_split_re, f.read_text())[:200]:
            if doc != "<|endoftext|>" and doc:
                n_bytes = len(doc.encode())
                # print(doc)
                toks = tok.encode(doc)
                # print(toks)
                n_tokens = len(toks)
                ratio = n_bytes / n_tokens
                ratios.append(ratio)
        avg_ratio = np.mean(ratios)
        print(f"{f.name} compression ratio: {avg_ratio:.2f}")


    tokenized_path = Path(args.tokenized_data_path)
    tokenized_path.mkdir(exist_ok=True, parents=True)
    fpaths = list(input_dir.glob("*-valid.txt"))
    # fpaths.extend(list(input_dir.glob("*-val.txt")))

    for fpath in fpaths:
        t0 = time.monotonic()
        tokens = tok.encode_file(fpath)
        taken = time.monotonic() - t0
        logger.info(f"Processed {str(fpath)}")
        logger.info(f"Took {taken:.1f} s.")
        logger.info(f"Throughput: {fpath.stat().st_size / (1024 * 1024) / taken:.2f} MB/s")
        fname = fpath.name
        np.save(str((tokenized_path / fname).with_suffix(".npy")), np.array(tokens, dtype="uint16"))