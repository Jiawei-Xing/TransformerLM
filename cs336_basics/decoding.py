import torch
import argparse
import json
from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.adamw import AdamW
from cs336_basics.tokenizer import Tokenizer

def decoder(x, model, max_token, vocab, temp, topp):
    for _ in range(max_token):
        # logits of the last token
        logits = model(x) # (seq_len, vocab_size)
        last_logits = logits[-1] # (vocab_size)

        # temperature scaling
        last_logits /= temp
        probs = softmax(last_logits, -1)

        # top-p sampling
        q = 0
        index = []
        probs_sorted, i_sorted = torch.sort(probs, descending=True)
        for p, i in zip(probs_sorted, i_sorted):
            q += p
            index.append(i)
            if q >= topp:
                break
        probs /= q

        # zero out the rest
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask[torch.tensor(index)] = True
        probs = probs.masked_fill(~mask, 0.0)

        # sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # concatenate as new sequence
        x = torch.cat([x, next_token])

        # stop if <endoftext>
        if vocab[next_token.item()] == b"<endoftext>":
            break

    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--merges", type=str, required=True)
    parser.add_argument("--max_token", type=int, default=200)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # init and load model
    transformer_lm = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    optimizer = AdamW(transformer_lm.parameters(), 1e-3, (0.9, 0.999), 1e-8, 0.01)
    load_checkpoint(args.checkpoint, transformer_lm, optimizer)

    # read vocab from file
    with open(args.vocab) as f1:
        raw_vocab = json.load(f1)
    vocab = {int(v): bytes(k, "latin1") for k, v in raw_vocab.items()}

    # encode init text
    tokenizer = Tokenizer.from_files(args.vocab, args.merges)
    init_tensor = torch.tensor(tokenizer.encode(args.init), device=device)

    # decode from transformer lm
    x = decoder(init_tensor, transformer_lm, args.max_token, vocab, args.temp, args.topp)
    x = x.cpu().tolist()
    text = tokenizer.decode(x)
    print(text)

