
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import shutil
import dataclasses
import fire
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

# from llama_recipes.utils import fsdp_auto_wrap_policy
from utils.fsdp_utils import fsdp_auto_wrap_policy, hsdp_device_mesh
from long_context_synthetic_evaluation import evaluate_medalign

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

from utils.dataset_utils import get_preprocessed_dataset

from utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)

from accelerate.utils import is_xpu_available

from llama_recipes.model_checkpointing import save_model_checkpoint

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.utils import resample
from typing import Callable, List, Tuple
import math


# Import evaluation function from eval.py
import lm_eval
from eval import *
import pandas as pd

############################################################

PROJECT_NAME = "TIMER"
SEED = 42
IGNORE_INDEX = -100
MAX_EPOCHS = 10000
CHKPT_DIR = "../models/distributed"


torch.set_printoptions(sci_mode=False)
torch.manual_seed(SEED)
random.seed(SEED)

############################################################

class CosineDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        t_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False,
    ) -> None:
        self.t_max = t_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):  # type: ignore
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos((self.last_epoch) * math.pi / self.t_max))
            / 2
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

class CosineWithWarmupAndLRScaling(torch.optim.lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer,
        max_iters: int,
        warmup_iters: int,
        min_lr: float = 0,
        last_epoch: int = -1,
    ) -> None:
        def lr_lambda(current_step: int) -> float:
            return current_step / warmup_iters

        multiplicative_lr = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch
        )

        cosine_lr = CosineDecay(
            optimizer, eta_min=min_lr, t_max=max_iters - warmup_iters, last_epoch=last_epoch
        )

        super().__init__(
            optimizer, [multiplicative_lr, cosine_lr], [warmup_iters], last_epoch
        )

        print('loaded scheduler')
        print('max_iters:', max_iters)
        print('warmup_iters:', warmup_iters)
        print('min_lr:', min_lr)

    def step(self, epoch=None) -> None:
        super().step()
        for idx, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = param_group.get("lr_scale", 1) * param_group["lr"]
            self._last_lr[idx] = param_group["lr"]
            print('lr:', param_group["lr"])

############################################################
def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    init_dict['entity'] = os.getenv("WANDB_ENTITY")
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)    
    return run

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    full_eval = kwargs.get("full_eval", None)
    gradient_accumulation_steps = kwargs.get("grad_accumulation_steps", None)
    print('Gradient Accumulation Steps:', gradient_accumulation_steps)
    print(kwargs)
    train_steps = kwargs.get("dataset_size", None)*kwargs.get("num_epochs", None)
    lr_warmup = train_steps*0.02
    print('Full eval with Medalign:', full_eval)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                # load_in_8bit=True if train_config.quantization else None,
                load_in_4bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        print('llama model loading')
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            # load_in_8bit=True if train_config.quantization else None,
            load_in_4bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
            # attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        try:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config()
        # Generate the peft config and start fine-tuning from original model
        except:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
        **kwargs,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="val",
        **kwargs,
    )

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
        **kwargs,
    )
    dataset_test.save_split(train_config.output_dir + "/test_split.json")
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")
    
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    
    # Create DataLoaders for the training and validation dataset

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

        test_dl_kwargs = get_dataloader_kwargs(train_config, dataset_test, tokenizer, "test")
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        if len(eval_dataloader) == 0:
            raise ValueError("The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set.")
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    try:
        if train_config.from_peft_checkpoint:
            optimizer_state = torch.load(train_config.from_peft_checkpoint + "/optimizer.pt")
            scheduler_state = torch.load(train_config.from_peft_checkpoint + "/scheduler.pt")
            scheduler = CosineWithWarmupAndLRScaling(
            optimizer,
            train_steps // gradient_accumulation_steps,
            lr_warmup // gradient_accumulation_steps,
            min_lr=0
        )
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)
            print("Loaded optimizer and scheduler states from checkpoint")
    except AttributeError:
        print("No checkpoint loaded for optimizer and scheduler")
        print('train_steps:', train_steps)
        print('gradient_accumulation_steps:', gradient_accumulation_steps)
        print('lr_warmup:', lr_warmup)
        scheduler = CosineWithWarmupAndLRScaling(
        optimizer,
        train_steps // gradient_accumulation_steps,
        lr_warmup // gradient_accumulation_steps,
        min_lr=0,
    )

    # Initialize variables to track best performance
    best_accuracy = 0
    best_epoch = -1
    all_results = []

    # Ensure the output directory exists
    os.makedirs(train_config.output_dir, exist_ok=True)
    MAX_EPOCHS = 15
    def eval_callback(epoch, model, train_config, optimizer, rank):
        nonlocal best_accuracy, best_epoch, all_results

        # Save the current model
        while True:
            current_model_path = os.path.join(train_config.output_dir, f"model_epoch_{epoch}")
            print(f"checking if epoch {epoch} exists", current_model_path)
            if not os.path.exists(current_model_path):
                break
            epoch += 1
            if epoch>=MAX_EPOCHS:
                break
        current_model_path = os.path.join(train_config.output_dir, f"model_epoch_{epoch}")
        if current_model_path is not None:
            os.makedirs(current_model_path, exist_ok=True)
        if not train_config.enable_fsdp or rank == 0:
            if train_config.use_peft:
                model.save_pretrained(current_model_path)
            else:
                save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch, output_dir=current_model_path)

        # Ensure all processes have finished saving
        if train_config.enable_fsdp:
            torch.distributed.barrier()


    def evaluate_MMLU(train_config, epoch):
        pretrain_str = f'pretrained={train_config.model_name}'
        # dtype_str = 'dtype="float"'
        peft_str = f'peft={os.path.join(train_config.output_dir, f"model_epoch_{epoch}")}'
        model_arg_str = str(pretrain_str+','+peft_str+',trust_remote_code=True')
        # model_arg_str = str(pretrain_str)+",trust_remote_code=True"
        print(f"model_arg_str: {model_arg_str}")

        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=model_arg_str,
            tasks=["mmlu"],
            batch_size=8,
            limit=2000,
        )
        return results

    def lm_eval_medalign(train_config, epoch):
        pretrain_str = f'pretrained={train_config.model_name}'
        peft_str = f'peft={os.path.join(train_config.output_dir, f"model_epoch_{epoch}")}'
        model_arg_str = str(pretrain_str+','+peft_str+',trust_remote_code=True')
        print(f"model_arg_str: {model_arg_str}")

        results = lm_eval.simple_evaluate(
            model='hf',
            model_args=model_arg_str,
            tasks=["medalign_evaluation"],
            batch_size=8,
            limit=50,
        )
        return results

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
        eval_callback=eval_callback
    )

if __name__ == "__main__":
    fire.Fire(main)
