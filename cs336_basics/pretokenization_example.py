import os
from typing import BinaryIO
from multiprocessing import Pool
import regex
from collections import Counter

from splitter import Splitter

GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pattern = regex.compile(GPT2_TOKENIZER_REGEX, regex.UNICODE)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def build_special_split_pattern(special_tokens: list[str]) -> regex.Pattern:
    escaped = [regex.escape(t) for t in special_tokens]
    return regex.compile("(?:%s)" % "|".join(escaped))


def split_with_regex(args) -> list[str]:
    filepath, start, end, special_token_split_pattern = args

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        segments = special_token_split_pattern.split(chunk)

        counter = Counter()

        for segment in segments:
            if not segment:
                continue
            for m in pattern.finditer(segment):
                token = m.group(0)
                counter[token] += 1
        return counter


def _seek_and_split_wrapper(args):
    splitter, filepath, start, end = args
    return splitter.seek_and_split(filepath, start, end)


def pre_tokenize(splitter: Splitter,
                 filepath: str, num_processes: int, special_token: str = "<|endoftext|>") -> dict[str, int]:
    """
    Pre-tokenize a large text file by chunking it into parts that can be
    processed independently.
    """

    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 4, special_token.encode())

    # special_token_split_pattern = build_special_split_pattern([special_token])
    pre_token_counts = {}
    chunk_args = [(filepath, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    args = [(splitter, fp, st, ed) for fp, st, ed in chunk_args]

    if num_processes > 1:
        with Pool(num_processes) as pool:
            result = pool.map(_seek_and_split_wrapper, args)
            for pre_token_counts_sample in result:
                for k, v in pre_token_counts_sample.items():
                    if k in pre_token_counts:
                        pre_token_counts[k] += v
                    else:
                        pre_token_counts[k] = v
    else:
        for fp, st, ed in chunk_args:
            pre_token_counts_sample = splitter.seek_and_split(fp, st, ed)
            for k, v in pre_token_counts_sample.items():
                if k in pre_token_counts:
                    pre_token_counts[k] += v
                else:
                    pre_token_counts[k] = v

    # final_result = Counter()
    # for chunk in result:
    #     final_result.update(chunk)
    #
    # return dict(final_result)
    return pre_token_counts



## Usage
with open("/home/katinska/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        print(f"Chunk from {start} to {end} has {len(chunk)} characters.")