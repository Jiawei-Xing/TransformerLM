from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
import multiprocessing as mp

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
    
    # bytes pairs to merge
    pairs = {}
    pairs_counts = {}
    for pretoken in all_pretoken_counts.keys():
        if len(pretoken) < 2:
            continue
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])

            # trace bytes pairs in the pretoken
            if pair not in pairs:
                pairs[pair] = {}
            if pretoken not in pairs[pair].keys():
                pairs[pair][pretoken] = []
            pairs[pair][pretoken].append(i)

            # count the pairs
            if pair not in pairs_counts:
                pairs_counts[pair] = 0
            pairs_counts[pair] += all_pretoken_counts[pretoken]

    # init vocab and merges
    vocab = [token.encode("utf-8") for token in special_tokens] + [bytes([i]) for i in range(256)]
    merges = []

    # merge pairs until vocab size is reached
    while len(vocab) < vocab_size:
        # break if no more pairs to merge
        if not pairs:
            break
        # merge the most common pair
        most_common_pair, _ = max(pairs_counts.items(), key=lambda x: (x[1], x[0]))
        merges.append(most_common_pair)

        # create new token from the pair
        new_token = most_common_pair[0] + most_common_pair[1]
        vocab.append(new_token)

        # update pretokens
        for pretoken in list(pairs[most_common_pair].keys()):
            positions = pairs[most_common_pair][pretoken]
            count = all_pretoken_counts.pop(pretoken)
            
            # merge the pair in the pretoken
            for i in positions:
                new_pretoken = pretoken[:i] + (new_token,) + pretoken[i+2:]
            all_pretoken_counts[new_pretoken] = all_pretoken_counts.get(new_pretoken, 0) + count

            # update pair positions containing the pretoken
            for pair in pairs:
                if pretoken in pairs[pair]:
                    old_pos = pairs[pair].pop(pretoken)
                    pairs[pair][new_pretoken] = [pos if pos <= i else pos - 1 for pos in old_pos]

        # update bytes pairs
        for pretoken, positions in pairs[most_common_pair].items():
            for pos in positions:
                # update left side of the pair
                if pos > 0:
                    # remove the old left pair
                    old_pair = (pretoken[pos-1], most_common_pair[0])
                    pairs[old_pair][pretoken].remove(pos-1)

                    # create a new left pair with the new token
                    new_pair = (pretoken[pos-1], new_token)
                    if new_pair not in pairs:
                        pairs[new_pair] = {}
                    if pretoken not in pairs[new_pair]:
                        pairs[new_pair][pretoken] = []
                    pairs[new_pair][pretoken].append(pos-1)

                # update right side of the pair
                if pos < len(pretoken) - 2:
                    # remove the old right pair
                    old_pair = (most_common_pair[1], pretoken[pos+1])
                    pairs[old_pair][pretoken].remove(pos)

                    # create a new right pair with the new token
                    new_pair = (new_token, pretoken[pos+1])
                    if new_pair not in pairs:
                        pairs[new_pair] = {}
                    if pretoken not in pairs[new_pair]:
                        pairs[new_pair][pretoken] = []
                    pairs[new_pair][pretoken].append(pos)

        # update the pairs counts
        pairs_counts = {}
        for pair in pairs:
            pairs_counts[pair] = sum(all_pretoken_counts[pretoken] * len(pos) for pretoken, pos in pairs[pair].items())

    # vocab mapping
    vocab_mapping = {i: v for i, v in enumerate(vocab)}

    return vocab_mapping, merges
