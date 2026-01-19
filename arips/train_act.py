
from pathlib import Path
from pprint import pformat
import logging

from termcolor import colored
import torch
from tqdm import tqdm
from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_train import train, update_policy
from lerobot.policies import ACTConfig
from datetime import datetime
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
import time

from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.train_utils import get_step_checkpoint_dir, get_step_identifier, save_checkpoint, update_last_checkpoint
from lerobot.utils.utils import format_big_number, init_logging

from torchvision.transforms import v2

BASE_REPO_ID = "pick_and_place_3seq"
TRAIN_REPO_ID = "jgdo/" + BASE_REPO_ID + "_train"
VAL_REPO_ID = "jgdo/" + "pick_and_place_3seq_val" # BASE_REPO_ID + "_val"


# dataset = LeRobotDataset("jgdo/pick_and_place_3seq_train")
# for frame in tqdm(dataset):
#     pass
# print("All frames loaded successfully.")
# exit(0)

dataset_config = DatasetConfig(
    repo_id=TRAIN_REPO_ID,
)

policy_config = ACTConfig(
    device="cuda",
    push_to_hub=False,
    chunk_size=15,
    n_action_steps=15,
)

now = datetime.now()
now_str = now.strftime("%Y-%m-%d_%H:%M:%S")

cfg = TrainPipelineConfig(
    dataset=dataset_config,
    policy=policy_config,
    output_dir=Path(f"outputs/act/{BASE_REPO_ID}_{now_str}"),
    job_name="act_affine_chunk_size_15",
    wandb=WandBConfig(enable=True),
    batch_size=40,
    num_workers=8,
    steps=5000,
    eval_freq=1000,
    save_freq=1000,
    log_freq=100,
)

def load_dataset(repo_id: str) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(
         repo_id
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=cfg.tolerance_s,
    )

    dataset = LeRobotDataset(
        repo_id=repo_id,
    )
    return dataset


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

init_logging(accelerator=accelerator)

cfg.validate()

logging.info(pformat(cfg.to_dict()))

wandb_logger = WandBLogger(cfg)

device = accelerator.device
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.info("Creating dataset")
train_dataset = make_dataset(cfg)


val_cfg = TrainPipelineConfig(
    dataset=DatasetConfig(
        repo_id=VAL_REPO_ID,
    ),
    policy=policy_config,
    wandb=WandBConfig(enable=False),
    batch_size=8,
    num_workers=4,
)

val_dataset = make_dataset(val_cfg) 

accelerator.wait_for_everyone()

logging.info("Creating policy")
policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset.meta,
        rename_map=cfg.rename_map,
    )

accelerator.wait_for_everyone()

# Create processors - only provide dataset_stats if not resuming from saved processors
processor_kwargs = {}
postprocessor_kwargs = {}
if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
# Only provide dataset_stats when not resuming from saved processor state
    processor_kwargs["dataset_stats"] = train_dataset.meta.stats

preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

logging.info("Creating optimizer and scheduler")
optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

step = 0  # number of policy updates (forward + backward + optim)
num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
num_total_params = sum(p.numel() for p in policy.parameters())

logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
logging.info(f"{train_dataset.num_frames=} ({format_big_number(train_dataset.num_frames)})")
logging.info(f"{train_dataset.num_episodes=}")
num_processes = accelerator.num_processes
effective_bs = cfg.batch_size * num_processes
logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

# create dataloader for offline training
if hasattr(cfg.policy, "drop_n_last_frames"):
    shuffle = False
    val_shuffle = False
    sampler = EpisodeAwareSampler(
        train_dataset.meta.episodes["dataset_from_index"],
        train_dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=train_dataset.episodes,
        drop_n_last_frames=cfg.policy.drop_n_last_frames,
        shuffle=True,
    )

    val_sampler = EpisodeAwareSampler(
            val_dataset.meta.episodes["dataset_from_index"],
            val_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=val_dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
    )
else:
    shuffle = True
    sampler = None
    val_shuffle = True
    val_sampler = None


dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=cfg.num_workers,
    batch_size=cfg.batch_size,
    shuffle=shuffle and not cfg.dataset.streaming,
    sampler=sampler,
    pin_memory=device.type == "cuda",
    drop_last=False,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    num_workers=cfg.num_workers,
    batch_size=cfg.batch_size,
    shuffle=val_shuffle,
    sampler=val_sampler,
    pin_memory=device.type == "cuda",
    drop_last=False,
    prefetch_factor=2 if cfg.num_workers > 0 else None,
)


# Prepare everything with accelerator
accelerator.wait_for_everyone()
policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    policy, optimizer, dataloader, lr_scheduler
)
dl_iter = cycle(dataloader)
#val_iter = cycle(val_dataloader)

policy.train()

train_metrics = {
    "loss": AverageMeter("loss", ":.3f"),
    "grad_norm": AverageMeter("grdn", ":.3f"),
    "lr": AverageMeter("lr", ":0.1e"),
    "update_s": AverageMeter("updt_s", ":.3f"),
    "dataloading_s": AverageMeter("data_s", ":.3f"),
}

# Use effective batch size for proper epoch calculation in distributed training
effective_batch_size = cfg.batch_size * accelerator.num_processes
train_tracker = MetricsTracker(
    effective_batch_size,
    train_dataset.num_frames,
    train_dataset.num_episodes,
    train_metrics,
    initial_step=step,
    accelerator=accelerator,
)

logging.info(
    f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
)

affine_transformer = v2.RandomAffine(degrees=(-8, 8), translate=(0.1, 0.1), scale=(0.9, 1.1))

for _ in range(step, cfg.steps):
    start_time = time.perf_counter()
    batch = next(dl_iter)
    batch["observation.images.top"] = affine_transformer(batch["observation.images.top"])
    batch = preprocessor(batch)
    train_tracker.dataloading_s = time.perf_counter() - start_time

    train_tracker, output_dict = update_policy(
        train_tracker,
        policy,
        batch,
        optimizer,
        cfg.optimizer.grad_clip_norm,
        accelerator=accelerator,
        lr_scheduler=lr_scheduler,
    )

    # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
    # increment `step` here.
    step += 1
    train_tracker.step()
    is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
    is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
    is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

    if is_log_step:
        logging.info(train_tracker)
        if wandb_logger:
            wandb_log_dict = train_tracker.to_dict()
            if output_dict:
                wandb_log_dict.update(output_dict)
            wandb_logger.log_dict(wandb_log_dict, step)
        train_tracker.reset_averages()

    if cfg.save_checkpoint and is_saving_step:
        logging.info(f"Checkpoint policy after step {step}")
        checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            cfg=cfg,
            policy=accelerator.unwrap_model(policy),
            optimizer=optimizer,
            scheduler=lr_scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )
        update_last_checkpoint(checkpoint_dir)
        if wandb_logger:
            wandb_logger.log_policy(checkpoint_dir)

        accelerator.wait_for_everyone()

    if is_eval_step:
        logging.info(f"Evaluating policy after step {step}")
        policy.eval()
        policy.config.use_vae = False
        val_loss = 0.0
        for eval_batch in val_dataloader:
            eval_batch = preprocessor(eval_batch)
            with torch.no_grad():
                batch_loss, _ = policy.forward(eval_batch)
            val_loss += batch_loss.item()
        val_loss /= len(val_dataloader)
        logging.info(f"Validation loss after step {step}: {val_loss:.4f}")
        if wandb_logger:
            wandb_logger.log_dict({"l1_loss": val_loss}, step, mode="eval")

        policy.config.use_vae = True

        policy.train()


logging.info("End of training")

if cfg.policy.push_to_hub:
    unwrapped_policy = accelerator.unwrap_model(policy)
    unwrapped_policy.push_model_to_hub(cfg)
    preprocessor.push_to_hub(cfg.policy.repo_id)
    postprocessor.push_to_hub(cfg.policy.repo_id)

# Properly clean up the distributed process group
accelerator.wait_for_everyone()
accelerator.end_training()
