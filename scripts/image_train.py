"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
import torch
import wandb


def main():
    args = create_argparser().parse_args()

    # Set up distributed training
    dist_util.init_distributed_mode(args)
    device = torch.device(args.device)

    # Specify the keys you want to include
    keys_to_include = {'image_size', 'diffusion_steps', 'num_res_blocks', 'num_channels', 'noise_schedule', 'lr', 'batch_size'}

    # Combine specified keys and values into a string
    identifier = '__'.join([f"{key}_{getattr(args, key)}" for key in vars(args) if key in keys_to_include])
    save_dir = f"{args.save_dir}/{identifier}"

    # Set up logging
    logger.configure(dir=save_dir)
    logger.log("creating model and diffusion...")    
    if dist_util.is_main_process():
        wandb.init(project="MaskDiff", config=args)

    # Create model and data
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        config=args,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        config=args,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=10000,
        save_dir="/scratch/as3ek/github/MaskDiff/saves",
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # Distributed training parameters
        device="cuda",
        seed=42,
        world_size=1,
        dist_url="env://",
        distributed=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
