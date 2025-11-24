import math
from collections import defaultdict
from dataclasses import dataclass
import tiktoken
from abc import ABC
import os
import sys
from splitter import Splitter

from pretokenization_example import pre_tokenize
from tests.conftest import vocab_size


class Tokenizer():
    """Abstract interface for a tokenizer."""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError
    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

def merge_key(left: int | bytes, right: int | bytes, k: tuple, new_id: int):
    new_k = list(k)
    i = 0
    updated_indices = []

    while i < len(new_k) - 1:
        if new_k[i] == left and new_k[i + 1] == right:
            new_k[i] = new_id
            new_k = new_k[:i + 1] + new_k[i + 2:] # drop here the right element
            updated_indices.append(i)
        i += 1
    return tuple(new_k), updated_indices




class BPETokenizer:
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, vocab_size, special_tokens):
        self.vocab_size: int = vocab_size
        self.special_tokens: list[str] = special_tokens
        self.special_token_bytes: list[bytes] = [st.encode() for st in special_tokens]
        self.merges: list[tuple[bytes | int, bytes | int]] = []
        self.merged_tuples: list[tuple[bytes, bytes]] = []
        self.vocab: dict[int, bytes] = {256 + i: st for i, st in enumerate(self.special_token_bytes)}
        self.new_id_to_bytes: dict[int, bytes] = self.vocab.copy()
        for i in range(256):
            self.vocab[i] = bytes([i])
        self.second_best_key = None
        self.splitter = Splitter("<|endoftext|>")


    @property
    def cur_vocab_size(self):
        """Current vocab size (different from self.vocab_size which is target vocab size)"""
        return len(self.vocab)

    def break_ties(self, sorted_all_counts):
        _, max_count = sorted_all_counts[0]
        ties = []
        candidates = []
        for i in range(len(sorted_all_counts)):
            key, count = sorted_all_counts[i]
            if count == max_count:
                ties.append(tuple(self.vocab[k] for k in key))
                candidates.append(key)
            else:
                break
        best_from_ties = max(ties)
        for i, cand in enumerate(candidates):
            if ties[i] == best_from_ties:
                max_key = cand
                break

        return max_key

    def merge_key(self, left: int | bytes, right: int | bytes, k: tuple, new_id: int):
        new_k = list(k)
        i = 0
        updated_indices = []

        while i < len(new_k) - 1:
            if new_k[i] == left and new_k[i + 1] == right:
                new_k[i] = new_id
                new_k = new_k[:i + 1] + new_k[i + 2:] # drop here the right element
                updated_indices.append(i)
            i += 1
        return tuple(new_k), updated_indices


    def convert(self, entry: int | bytes) -> bytes:
        if entry in self.vocab:
            return self.vocab[entry]
        bytestr: list[bytes] = self.new_id_to_bytes.get(entry, [entry])
        while any(elem in self.new_id_to_bytes for elem in bytestr):
            i = 0
            while i < len(bytestr):
                if bytestr[i] in self.new_id_to_bytes:
                    bytestr = bytestr[:i] + self.new_id_to_bytes[bytestr[i]] + bytestr[i + 1:]
                i += 1
        else:
            return bytes(bytestr)

    def update_counts(self, pre_token_byte_counts: dict[tuple[bytes], int],
                      pair_to_pre_tokens: dict[tuple[bytes, set]],
                      all_counts: dict | None = None):
        if all_counts is None:
            all_counts = defaultdict(int)
        else:
            all_counts = defaultdict(int, all_counts)
        for token_bytes, count in pre_token_byte_counts.items():
            for i in range(len(token_bytes) - 1):
                pair = (token_bytes[i], token_bytes[i + 1])
                all_counts[pair] += count
                if pair_to_pre_tokens is not None:
                    pair_to_pre_tokens.setdefault(pair, set()).add(token_bytes)
        return all_counts

    def merge_bpe(self, pre_token_byte_counts: dict[tuple[bytes], int]):
        counts: dict[tuple[bytes], int] = defaultdict(int)

        for token_bytes, count in pre_token_byte_counts.items():
            for i in range(len(token_bytes) - 1):
                pair = (token_bytes[i], token_bytes[i + 1])
                counts[pair] += count

        # print(counts)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_counts)
        max_key = max(sorted_counts, key=lambda x: x[1])[0]
        # print(max_key)
        new_id = self.cur_vocab_size
        new_pre_token_byte_counts = pre_token_byte_counts.copy()

        for k, v in pre_token_byte_counts.items():
            new_k, updated_indices = self.merge_key(max_key[0], max_key[1], k, new_id)
            if len(updated_indices):
                del new_pre_token_byte_counts[k]
                new_pre_token_byte_counts[new_k] = v

        return (max_key, new_id), new_pre_token_byte_counts

    def iter_merge(
            self,
            pre_token_byte_counts: dict[tuple[bytes], int],
    ) -> tuple[tuple[tuple[bytes], int], dict]:

        all_counts: dict[tuple[bytes], int] = defaultdict(int)
        for k, v in pre_token_byte_counts.items():
            for i in range(len(k) - 1):
                all_counts[(k[i], k[i + 1])] += v


        sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
        max_key = self.break_ties(sorted_all_counts)

        new_id = self.cur_vocab_size

        new_pre_token_byte_counts = pre_token_byte_counts.copy()

        left, right = max_key
        for k, v in pre_token_byte_counts.items():
            new_k, updated = self.merge_key(left, right, k, new_id)
            if updated:
                v = new_pre_token_byte_counts.pop(k)
                # counts are not changed, we just need to re-write with a new key
                new_pre_token_byte_counts[new_k] = v

        return (
            (max_key, new_id),
            new_pre_token_byte_counts,
        )

    def merge_bpe_cached(self, pre_token_byte_counts: dict[tuple[bytes], int],
                         updated_keys=None, all_counts=None, pair_to_pre_tokens=None, all_updated_pairs=None):

        if pair_to_pre_tokens is None:
            pair_to_pre_tokens = {}
            all_counts = self.update_counts(pre_token_byte_counts, pair_to_pre_tokens)
            all_updated_pairs = set(all_counts.keys())
        else:
            all_counts = self.update_counts({k: pre_token_byte_counts[k] for k in updated_keys},
                                            pair_to_pre_tokens, all_counts)

        # print(all_updated_pairs)
        # print("SECOND", self.second_best_key)
        if self.second_best_key is not None and all_updated_pairs:
            sorted_subset = sorted(
                [(k, all_counts[k]) for k in set([k for (k, v) in self.second_best_key]).union(all_updated_pairs)
                 if k in all_counts],
                key=lambda x: x[1], reverse=True)
            self.second_best_key = sorted_subset
            # print('SORTED', sorted_subset)
            max_key = self.break_ties(sorted_subset)
        else:
            sorted_all_counts = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
            counts_to_keep = math.ceil(len(sorted_all_counts) * 0.1)
            self.second_best_key = sorted_all_counts[:counts_to_keep]
            max_key = self.break_ties(sorted_all_counts)

        new_id = self.cur_vocab_size
        affected_pre_tokens: set[tuple[bytes]] = set()
        for (left, right), keys in pair_to_pre_tokens.items():
            if left in max_key and right in max_key:
                affected_pre_tokens.update(keys)
        # for pair, keys in pair_to_pre_tokens.items():
        #     if pair == max_key:
        #         affected_pre_tokens.update(keys)

        new_pre_token_byte_counts = pre_token_byte_counts.copy()
        new_updated_keys = set() #?
        all_counts_updated = all_counts.copy()
        pair_to_pre_tokens_updated = pair_to_pre_tokens.copy()
        all_updated_pairs_updated = set() # ?


        for k in affected_pre_tokens:
            new_k, updated_indices = self.merge_key(max_key[0], max_key[1], k, new_id)
            if updated_indices:
                v = new_pre_token_byte_counts.pop(k)
                # print('K', k, new_k)
                for pair in zip(k[:-1], k[1:]):
                    # print(pair, all_counts_updated[pair])
                    all_counts_updated[tuple(pair)] -= v
                    assert all_counts_updated[tuple(pair)] >= 0
                    # print('PAIR TO PRE: ', pair_to_pre_tokens_updated[pair])
                    pair_to_pre_tokens_updated[tuple(pair)].discard(k)
                    # print('PAIR TO PRE 2: ', pair_to_pre_tokens_updated[pair])

                new_pre_token_byte_counts[new_k] = v
                new_updated_keys.add(new_k)

                for j in updated_indices:
                    if j > 0:
                        left_pair = (new_k[j - 1], new_k[j])
                        all_updated_pairs_updated.add(left_pair)
                    if j < len(new_k) - 1:
                        right_pair = (new_k[j], new_k[j + 1])
                        all_updated_pairs_updated.add(right_pair)



        return(
            (max_key, new_id),
            new_pre_token_byte_counts, new_updated_keys,
            {k:v for k, v in all_counts_updated.items() if v > 0},
            pair_to_pre_tokens_updated, all_updated_pairs_updated
        )


    def train_bpe(self, input_path: str | os.PathLike):

        pre_counts = pre_tokenize(self.splitter, input_path, num_processes=4)

        self.pre_token_byte_counts: dict[tuple[bytes], int] = {
            tuple(v.encode()): c for v, c in pre_counts.items()
        }
        # print("Pre-token byte counts:", self.pre_token_byte_counts)

        n_iters = max(0, self.vocab_size - self.cur_vocab_size)

        # # Non-efficient implementation
        # for i in range(n_iters):
        #     (updated_key, new_id), self.pre_token_byte_counts = self.iter_merge(self.pre_token_byte_counts)
        #     self.new_id_to_bytes[new_id] = updated_key
        #     v = self.convert(new_id)
        #     self.merges.append(updated_key)
        #     converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
        #     self.merged_tuples.append(converted)
        #     self.vocab[new_id] = v


        updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = None, None, None, None

        for _ in range(n_iters):
            (
                (updated_key, new_id),
                self.pre_token_byte_counts, new_updated_keys,
                all_counts_updated, pair_to_pre_tokens_updated,
                all_updated_pairs_updated
            ) = self.merge_bpe_cached(
                self.pre_token_byte_counts,
                updated_keys,
                all_counts,
                pair_to_pre_tokens,
                all_updated_pairs
            )

            self.new_id_to_bytes[new_id] = updated_key
            self.merges.append(updated_key)
            v = self.convert(new_id)
            converted = (self.convert(updated_key[0]), self.convert(updated_key[1]))
            self.merged_tuples.append(converted)
            self.vocab[new_id] = v
            updated_keys, all_counts, pair_to_pre_tokens, all_updated_pairs = (new_updated_keys,
                                                                               all_counts_updated,
                                                                               pair_to_pre_tokens_updated,
                                                                               all_updated_pairs_updated)

        #
        # print(self.merges)
        return self.vocab, self.merged_tuples

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string



def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")





def train_bpe_simple(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1
        # Find the most common pair.
        pair = max(counts, key=counts.get)
        index1, index2 = pair
        # Merge that pair.
        new_index = 256 + i
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab=vocab, merges=merges)



def main():
    text_path = "/home/katinska/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    bpe = BPETokenizer(vocab_size, special_tokens=special_tokens)
    vocab, merges = bpe.train_bpe(text_path)
    with open("bpe-vocab.txt", "w", encoding="utf-8") as f:
        for index in sorted(vocab.keys()):
            token_bytes = vocab[index]
            f.write(f"{index}\t{token_bytes}\n")
    with open("bpe-merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left}\t{right}\n")



if __name__ == "__main__":
    main()