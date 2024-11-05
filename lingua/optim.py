# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from functools import partial
import math

import logging
from torch import nn
from torch.optim import AdamW, lr_scheduler

logger = logging.getLogger()


@dataclass
class OptimArgs:
    lr: float = 3e-4
    weight_decay: float = 0.1
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.95
    clip: float = 1.0

    scheduler: str = "cosine"
    warmup: int = 2000
    lr_min_ratio: float = 0.1
    cycle_length: float = 1.0
    cosine_theta: float = 1.0
    annealing_step: int = 1000

    exp_factor: float = 0.5


def lr_linear(step: int, warmup: int, n_steps: int, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = s * min_ratio + (1 - s)
    else:
        lr = min_ratio
    return lr


def lr_inv_sqrt(step: int, warmup: int, exp_factor: float, min_ratio: float) -> float:
    if step < warmup:
        lr = float(step) / warmup
    else:
        lr = max((warmup**exp_factor) / (step**exp_factor), min_ratio)
    return lr


def lr_cosine(
    step: int,
    warmup: int,
    n_steps: int,
    cycle_length: float,
    theta: float,
    min_ratio: float,
) -> float:
    if step < warmup:
        lr = float(step) / warmup
    elif step <= n_steps:
        s = float(step - warmup) / (n_steps - warmup)
        lr = min_ratio + 0.5 * (1 - min_ratio) * (
            math.cos(math.pi * s**theta / cycle_length) + 1
        )
    else:
        lr = min_ratio
    return lr


def build_lr_fn(args: OptimArgs, n_steps: int):
    if args.scheduler == "constant":
        lr_fn = lambda x: 1.0
    elif args.scheduler == "linear":
        lr_fn = partial(
            lr_linear, warmup=args.warmup, n_steps=n_steps, min_ratio=args.lr_min_ratio
        )
    elif args.scheduler == "inv_sqrt":
        lr_fn = partial(
            lr_inv_sqrt,
            warmup=args.warmup,
            exp_factor=args.exp_factor,
            min_ratio=args.lr_min_ratio,
        )
    elif args.scheduler == "cosine":
        lr_fn = partial(
            lr_cosine,
            warmup=args.warmup,
            n_steps=n_steps,
            cycle_length=args.cycle_length,
            theta=args.cosine_theta,
            min_ratio=args.lr_min_ratio,
        )
    else:
        raise NotImplementedError(f"Unknown scheduler: {args.scheduler}")
    return lr_fn


def build_optimizer(model: nn.Module, args: OptimArgs, n_steps: int):
    logger.info("Starting build of optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.epsilon,
        fused=True,  # Faster optim.step but can throw errors
    )

    # scheduler
    lr_fn = build_lr_fn(args, n_steps)
    scheduler = lr_scheduler.LambdaLR(
        optimizer, lr_fn
    )  # lr_scheduler.LambdaLR(optimizer, lr_fn)

    logger.info("Done with build of optimizer.")
    return optimizer, scheduler
