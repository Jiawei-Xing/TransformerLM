import math

def cosine_annealing(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + \
            0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) \
            * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate