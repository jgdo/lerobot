
import logging

import torch
from tqdm import tqdm
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from lerobot.utils.utils import init_logging


REPO_ID = "jgdo/pick_and_place_3seq_val"
MODEL = "outputs/act/jgdo/pick_and_place_3seq_2026-01-17_18:38:19"
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
