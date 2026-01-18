
import os
import time
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.processor.core import RobotAction
from lerobot.processor.factory import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.scripts.lerobot_record import record_loop
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 30*30


def main():
    model_id = "/home/jgdo/dev/lerobot/outputs/train/act_pick_place_all_train/checkpoints/last/pretrained_model"
    policy = ACTPolicy.from_pretrained(model_id)

    # # find ports using lerobot-find-port
    follower_port = "/dev/ttyUSB0"  # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    follower_id = "my_lerobot"  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
        "top": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=30),
    }

    robot_cfg = SO101FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config, use_degrees=False)
    robot = SO101Follower(robot_cfg)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()


    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            )
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features)
        ),
    )
    
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_id,
        dataset_stats={},
        preprocessor_overrides={
                    "device_processor": {"device": "cuda"},
                    "rename_observations_processor": {"rename_map": {}},
                },
    )

    robot.connect()

    fps =30
    events={"exit_early": False}
    control_time_s=300
    single_task="Pick and place object"
    display_data=True

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

        # Get action from either policy or teleop
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        act_processed_policy: RobotAction = make_robot_action(action_values, dataset_features)


        # Applies a pipeline to the action, default is IdentityProcessor
        
        action_values = act_processed_policy
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))
      

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


if __name__ == "__main__":
    main()
