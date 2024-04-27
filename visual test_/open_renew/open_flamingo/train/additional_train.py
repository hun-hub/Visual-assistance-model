
import argparse
import glob
import os
import random

import numpy as np
import torch
import wandb
from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from train_utils import (
    train_one_epoch,
    get_mp_policy_dtype,
    save_checkpoint,
)
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
from torch.distributed.distributed_c10d import _get_default_group
import functools

from open_flamingo import create_model_and_transforms


from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp._init_utils import _init_intra_and_inter_node_groups
import functools
from open_flamingo import create_model_and_transforms

def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_model(args, model, tokenizer):
    # Set up distributed training
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    random_seed(args.seed)

    # Initialize logging
    print(f"Start running training on rank {args.rank}.")
    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    # Load model checkpoint on CPU
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        # if args do not specify a checkpoint to resume from, check if checkpoints exist for this run
        # and automatically resume from the latest checkpoint
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}."
            )

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        msd = checkpoint["model_state_dict"]
        msd = {k.replace("module.", ""): v for k, v in msd.items()}
        resume_from_epoch = checkpoint["epoch"] + 1

        # for fsdp, only one rank needs to load the state dict
        if not args.fsdp or args.rank == 0:
            model.load_state_dict(msd, False)

    # Initialize FSDP / DDP, and ensure the model is on GPU
    print(f"Initializing distributed training with {args.world_size} GPUs.")
    if args.fsdp:
        print(
            f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )

        # init MixedPrecision
        if args.precision != "fp32":
            cast_dtype = get_mp_policy_dtype(args.precision)
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=cast_dtype,  # gradient communication
                buffer_dtype=cast_dtype,
            )
        else:
            mp_policy = None

        # init process groups
        if args.fsdp_sharding_strategy == "hybrid":
            intra_node_group, inter_node_group = _init_intra_and_inter_node_groups(
                _get_default_group()
            )
            args.my_group = intra_node_group  # for optimizer saving
            process_group = (intra_node_group, inter_node_group)  # for FSDP init
        else:
            args.my_group = None  # for optimizer saving
            process_group = None  # for FSDP init

        # init FSDP
        wrapper_kwargs = dict(
            process_group=process_group,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=device_id,
            sync_module_states=True,  # broadcast loaded ckpt from rank 0 -> all ranks
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if args.fsdp_sharding_strategy == "full"
            else ShardingStrategy.HYBRID_SHARD,
            use_orig_params=args.fsdp_use_orig_params,
            mixed_precision=mp_policy,
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
        )
        model.wrap_fsdp(wrapper_kwargs, device_id)
        ddp_model = model

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters())} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    else:
        model = model.to(device_id)
        ddp_model = DDP(model, device_ids=[device_id])

    # Initialize gradient checkpointing
    if args.gradient_checkpointing:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            offload_to_cpu=True,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            ddp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda m: getattr(m, "_use_gradient_checkpointing", False)
            and not isinstance(m, FSDP)
            and not isinstance(m, CheckpointWrapper),
        )

    # Initialize optimizer
    params_to_optimize = ddp_model.named_parameters()
    params_to_optimize = list(
        filter(
            lambda x: x[1].requires_grad
            and not getattr(x[1], "exclude_from_optimizer", False),
            params_to_optimize,
        )
    )
    if not args.fsdp or args.fsdp_use_orig_params:
        # apply weight decay only to params in the xattn layers
        def get_grouped_params(model):
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)
            return [
                {"params": params_with_wd, "weight_decay": args.weight_decay},
                {"params": params_without_wd, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            get_grouped_params(params_to_optimize), lr=args.learning_rate
        )
    else:
        optimizer = torch.optim.AdamW(
        (p for _, p in params_to_optimize),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        )
        # unclear if we should be using no weight decay or small weight