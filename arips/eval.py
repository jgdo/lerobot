from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.robots import RobotConfig, Robot
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.policies import ACTConfig

import os

robot_config = SO101FollowerConfig(
    port="/dev/ttyUSB0",
    id="my_lerobot",
    use_degrees=False,
    cameras={
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        "top": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=30),
    },
)

dataset_config = DatasetRecordConfig(
    repo_id="jgdo/eval_act",
    single_task="Pick and place object",
    push_to_hub=False,
    episode_time_s=300,
)


policy_config = PreTrainedConfig.from_pretrained(
     "/home/jgdo/dev/lerobot/outputs/train/act_pick_and_place_3seq/checkpoints/last/pretrained_model"
     )
policy_config.pretrained_path = "/home/jgdo/dev/lerobot/outputs/train/act_pick_and_place_3seq/checkpoints/last/pretrained_model"


record_config = RecordConfig(
    robot=robot_config,
    display_data=True,
    dataset=dataset_config,
    policy=policy_config,
)

os.system("rm -rf /home/jgdo/.cache/huggingface/lerobot/jgdo/eval_act")

record(record_config)
