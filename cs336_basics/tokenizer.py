import json
import regex as re
import numpy as np
from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        
        # add special tokens to vocab
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token_bytes
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary (json)
        and list of merges (txt) and optionally a list of special tokens. 
        """
        # read vocab from file
        with open(vocab_filepath) as f1:
            raw_vocab = json.load(f1)
        vocab = {int(v): bytes(k, "latin1") for k, v in raw_vocab.items()}

        # read merges from file
        merges = []
        with open(merges_filepath) as f2:
            for line in f2:
                line = line.rstrip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    m = line.split(" ", -1)
                    if len(m) == 2:  # Ensure we have exactly 2 parts
                        merges.append((bytes(m[0], "latin1"), bytes(m[1], "latin1")))

        return cls(vocab, merges, special_tokens)
        
    def encode1(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs, restarting merges every time.
        """
        # pretokenization by special tokens
        if self.special_tokens:
            # Sort special tokens by length (descending) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
            splits = re.split(f"({pattern})", text)  # Added parentheses to preserve delimiters
        else:
            splits = [text]
        
        # vocab IDs mapping
        vocab_ID = {b: i for i, b in self.vocab.items()}
        IDs = []
        for split in splits:
            if not split: # skip empty splits
                continue
                
            # directly add ID if split is a special token
            if self.special_tokens and split in self.special_tokens:
                special_bytes = split.encode("utf-8")
                IDs.append(vocab_ID[special_bytes])
                continue

            # pretokenization by regex
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, split):
                pretoken = match.group(0)

                # pretoken string to bytes
                pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
                
                while len(pretoken_bytes) > 1:
                    # find the first pair to merge
                    pairs = [(pretoken_bytes[i], pretoken_bytes[i+1]) for i in range(len(pretoken_bytes)-1)]
                    merge = [m for m in self.merges if m in pairs]
                    if not merge:
                        break
                    merge = merge[0]
        
                    new_token = []
                    i = 0
                    # merge pairs in pretoken_bytes
                    while i < len(pretoken_bytes):
                        if i < len(pretoken_bytes)-1: # not the last byte
                            if (pretoken_bytes[i], pretoken_bytes[i+1]) == merge:
                                new_token.append(merge[0] + merge[1]) # merge
                                i += 2
                                continue
                        # not merge
                        new_token.append(pretoken_bytes[i])
                        i += 1
                        
                    # update pretoken bytes after each merge
                    pretoken_bytes = new_token
            
                # convert final protoken bytes to IDs
                IDs += [vocab_ID[b] for b in pretoken_bytes]

        return IDs

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs, but does not restart merges.
        """
        # pretokenization by special tokens
        if self.special_tokens:
            # Sort special tokens by length (descending) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
            splits = re.split(f"({pattern})", text)  # Added parentheses to preserve delimiters
        else:
            splits = [text]
        
        # vocab IDs mapping
        vocab_ID = {b: i for i, b in self.vocab.items()}
        IDs = []
        for split in splits:
            if not split: # skip empty splits
                continue
                
            # directly add ID if split is a special token
            if self.special_tokens and split in self.special_tokens:
                special_bytes = split.encode("utf-8")
                IDs.append(vocab_ID[special_bytes])
                continue

            # pretokenization by regex
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in re.finditer(PAT, split):
                pretoken = match.group(0)

                # pretoken string to bytes
                pretoken_bytes = [bytes([b]) for b in pretoken.encode("utf-8")]
                
                for merge in self.merges:
                    new_token = []
                    i = 0
                    # merge pairs in pretoken_bytes
                    while i < len(pretoken_bytes):
                        if i < len(pretoken_bytes)-1: # not the last byte
                            if (pretoken_bytes[i], pretoken_bytes[i+1]) == merge:
                                new_token.append(merge[0] + merge[1]) # merge
                                i += 2
                                continue
                        # not merge
                        new_token.append(pretoken_bytes[i])
                        i += 1
                        
                    # update pretoken bytes after each merge
                    pretoken_bytes = new_token
            
                # convert final protoken bytes to IDs
                IDs += [vocab_ID[b] for b in pretoken_bytes]

        return IDs
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator 
        that lazily yields token IDs. This is required for memory-efficient tokenization 
        of large files that we cannot directly load into memory.
        """
        for each in iterable:
            yield from self.encode(each)
            
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        # first concatenate all byte sequences!
        byte_sequence = b"".join(self.vocab[ID] for ID in ids)
        
        # then decode string with error handling
        result = byte_sequence.decode("utf-8", errors="replace")

        return result

if __name__ == "__main__":
    special_tokens = ["<endoftext>"]
    tokenizer = Tokenizer.from_files("../results/vocab.json", "../results/merges.txt", special_tokens)
    with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        text = f.read()
    IDs = tokenizer.encode(text)
    np.save("../results/tokens.npy", np.asarray(IDs))

