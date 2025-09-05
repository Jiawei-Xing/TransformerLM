from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
import multiprocessing as mp
import json

# Parallelize pretoken counting using multiprocessing
def process_chunk(args):
    input_path, start, end, special_tokens = args

    # read file by chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Split the chunk into pre-tokens based on special tokens
    splits = re.split("|".join(re.escape(tok) for tok in special_tokens), chunk)

    # further split by regex
    pretokens = []
    for split in splits:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens += [n.group(0) for n in re.finditer(PAT, split)]

    # Count occurrences of each pre-token
    pretoken_counts = Counter(pretokens)

    # Convert pre-tokens to byte tuples
    byte_counts = {tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in pretoken_counts.items()}

    return byte_counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) model on the input file.
    Args:
        input_path (str): Path to the input file.
        vocab_size (int): Desired size of the vocabulary.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.
    Returns:
        vocab: a dictionary mapping token IDs to byte strings.
        merges: a list of pairs of byte strings representing the BPE merges.
    """
    # chunk boundaries
    num_processes = mp.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))

    # multiprocessing to process chunks
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunk_args)

    # Combine counts from all processes
    all_pretoken_counts = Counter()
    for c in results:
        all_pretoken_counts.update(c)
    
    # init vocab and merges
    vocab = {}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    merges = []

    # merge pairs until vocab size is reached
    while len(vocab) < vocab_size:
        # Count all pairs in current pretokens
        pairs_counts = {}
        for pretoken, count in all_pretoken_counts.items():
            if len(pretoken) < 2:
                continue
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i + 1])
                pairs_counts[pair] = pairs_counts.get(pair, 0) + count
        
        # break if no more pairs to merge
        if not pairs_counts:
            break
            
        # merge the most common pair
        most_common_pair = max(pairs_counts, key=lambda x: (pairs_counts[x], x))
        merges.append(most_common_pair)

        # create new token from the pair
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab[len(vocab)] = new_token

        # update pretokens by merging the most common pair
        new_pretoken_counts = {}
        for pretoken, count in all_pretoken_counts.items():
            if len(pretoken) < 2: # single byte
                new_pretoken_counts[pretoken] = new_pretoken_counts.get(pretoken, 0) + count
                continue
                
            # Apply merges to this pretoken
            new_pretoken = []
            i = 0
            while i < len(pretoken):
                if i < len(pretoken) - 1: # not the last byte
                    if (pretoken[i], pretoken[i + 1]) == most_common_pair:
                        new_pretoken.append(new_token) # merge
                        i += 2  # Skip the next token since we merged
                        continue
                # not merge
                new_pretoken.append(pretoken[i])
                i += 1
            
            new_pretoken_counts[tuple(new_pretoken)] = new_pretoken_counts.get(pretoken, 0) + count
        
        all_pretoken_counts = new_pretoken_counts

    return vocab, merges

if __name__ == "__main__":
    special_tokens = ["<endoftext>"]
    vocab, merges = train_bpe("../data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)

    # Save vocab as JSON (id -> string with latin1 so it’s reversible)
    with open("../results/vocab.json", "w", encoding="utf-8") as f:
        json.dump({v.decode("latin1"): k for k, v in vocab.items()},
                f, ensure_ascii=False, indent=2)

    # Save merges as TXT (two tokens per line, latin1-safe)
    with open("../results/merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")

