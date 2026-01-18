from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.utils.robot_utils import precise_sleep
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

import time

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 300

MODEL = "outputs/train/act_pick_place_all_train"
CHECKPOINT = "last"

def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = f"{MODEL}/checkpoints/{CHECKPOINT}/pretrained_model"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "jgdo/pick_place_all"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    # # find ports using lerobot-find-port
    follower_port = "/dev/ttyUSB0" 

    # # the robot ids are used the load the right calibration files
    follower_id = 'my_lerobot'

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        "top": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=30),
    }

    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)
    robot.connect()

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            t0 = time.perf_counter()
            obs = robot.get_observation()
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)

            action = make_robot_action(action, dataset_metadata.features)

            robot.send_action(action)
            precise_sleep(max(1.0 / 30 - (time.perf_counter() - t0), 0.0))

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()