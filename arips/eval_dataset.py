
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
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.act.modeling_act import ACTPolicy
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


REPO_ID = "jgdo/pick_and_place_3seq_val"
MODEL = "outputs/train/act_pick_place_all_train"
CHECKPOINT = "last"
MODEL_ID = f"{MODEL}/checkpoints/{CHECKPOINT}/pretrained_model"


init_logging()

device = torch.device("cuda")

logging.info("Creating policy")
policy = ACTPolicy.from_pretrained(MODEL_ID)
policy_cfg = PreTrainedConfig.from_pretrained(MODEL_ID)


def load_dataset(repo_id: str) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(
         repo_id
    )
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-4,
    )
    return dataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.info("Creating dataset")
dataset = load_dataset(REPO_ID)

# Create processors
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg,
    pretrained_path=MODEL_ID,
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=8,
    shuffle=None,
    sampler=None,
    pin_memory=device.type == "cuda",
    drop_last=False,
    prefetch_factor=2,
)

logging.info(f"Evaluating policy")
policy.eval()
policy.config.use_vae = False
l1_loss = 0.0
for eval_batch in tqdm(dataloader):
    eval_batch = preprocessor(eval_batch)
    with torch.no_grad():
        batch_loss, _ = policy.forward(eval_batch)
    l1_loss += batch_loss.item()
l1_loss /= len(dataloader)
logging.info(f"Validation l1_loss: {l1_loss:.4f}")
