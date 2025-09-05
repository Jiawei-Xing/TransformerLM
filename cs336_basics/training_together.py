import torch
import numpy as np
import argparse
import wandb
import os
from cs336_basics.data_loading import data_load
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.learning_rate_schedule import cosine_annealing
from cs336_basics.gradient_clipping import gradient_clipping

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--num_heads", type=int, default=16)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--rope_theta", type=float, default=10000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--betas", type=float, nargs=2, default=(0.9,0.999))
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--max_step", type=int, default=5000)
parser.add_argument("--max_l2_norm", type=float, default=1.0)
parser.add_argument("--min_learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_iters", type=int, default=500)
parser.add_argument("--cosine_cycle_iters", type=int, default=5000)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="cs336", name="training_together")

# loading data
input_data = np.load(args.input, mmap_mode="r")

# init model
transformer_lm = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)

# init optimizer
optimizer = AdamW(transformer_lm.parameters(), args.lr, args.betas, args.eps, args.weight_decay)

# load checkpoint
if os.path.exists(args.checkpoint):
    iteration = load_checkpoint(args.checkpoint, transformer_lm, optimizer)
else:
    iteration = 0

# training model
for t in range(iteration, args.max_step):
    # sample a batch
    batch_data = data_load(input_data, args.batch_size, args.context_length, device)

    # forward
    optimizer.zero_grad()
    logits = transformer_lm(batch_data[0])
    loss = cross_entropy(logits, batch_data[1])

    # backprop
    loss.backward()
    gradient_clipping(transformer_lm.parameters(), args.max_l2_norm)
    optimizer.step()

    # learning rate schedule
    lr = cosine_annealing(t, args.lr, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # checkpoint
    if t % 100 == 0:
        save_checkpoint(transformer_lm, optimizer, t, args.checkpoint)
    wandb.log({"loss": loss.item(), "learning_rate": lr, "iteration": t})

