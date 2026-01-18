"""
You can also use the CLI to record data. To see the required arguments, run:
lerobot-record --help
"""
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.processor.core import RobotAction, RobotObservation
from lerobot.processor.factory import make_default_robot_action_processor, make_default_robot_observation_processor
from lerobot.processor.pipeline import RobotProcessorPipeline
from lerobot.robots.so100_follower.ego_kinematic_processor import RobotEgoKinematicProcessor
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsEEToJoints
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SO101Follower
from lerobot.teleoperators.gamepad.teleop_gamepad import GamepadEgoTeleop, GamepadEgoTeleopConfig
from lerobot.teleoperators.so101_poti.config_so101_poti import SO101PotiConfig
from lerobot.teleoperators.so101_poti.so101_poti import SO101Poti
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.scripts.lerobot_record import record_loop

from time import gmtime, sleep, strftime

NUM_EPISODES = 50
FPS = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC = 2
TASK_DESCRIPTION = "pick the cube and place it into the box"

current_time_str = strftime("%Y-%m-%d_%H%M%S", gmtime())

HF_USER = "jgdo"

follower_port = "/dev/ttyUSB0"
follower_id = "my_lerobot"


# Create the robot and teleoperator configurations
camera_config = {
    "front": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
     "top": OpenCVCameraConfig(index_or_path=0, width=640, height=360, fps=FPS),
}
robot_config = SO101FollowerConfig(
    port=follower_port,
    id=follower_id,
    cameras=camera_config,
    use_degrees=False,
)

# Initialize the robot and teleoperator
robot = SO101Follower(robot_config)

# teleop_config = GamepadEgoTeleopConfig()
# teleop = GamepadEgoTeleop(teleop_config)
teleop_config = SO101PotiConfig(
        port="/dev/ttyACM0",
        motor_names=list(robot.bus.motors.keys()),
)
teleop = SO101Poti(teleop_config)


# NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
kinematics_solver = RobotKinematics(
    urdf_path="/home/jgdo/dev/SO-ARM100/Simulation/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert teleop action to joint action
teleop_action_processor = RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
](
    steps=[
        # RobotEgoKinematicProcessor(
        #     kinematics=kinematics_solver,
        #     end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
        #     motor_names=list(robot.bus.motors.keys()),
        #     use_latched_reference=True,
        # ),
        # InverseKinematicsEEToJoints(
        #     kinematics=kinematics_solver,
        #     motor_names=list(robot.bus.motors.keys()),
        #     initial_guess_current_joints=True,
        # ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# # Build pipeline to convert EE action to joints action
# robot_ee_to_joints_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
#     steps=[
#         InverseKinematicsEEToJoints(
#             kinematics=kinematics_solver,
#             motor_names=list(robot.bus.motors.keys()),
#             initial_guess_current_joints=True,
#         ),
#     ],
#     to_transition=robot_action_observation_to_transition,
#     to_output=transition_to_robot_action,
# )

robot_action_processor = make_default_robot_action_processor()
robot_observation_processor = make_default_robot_observation_processor()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset where to store the data
dataset = LeRobotDataset.create(
    repo_id=f"{HF_USER}/robot-learning-tutorial-data-{current_time_str}",
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

# Initialize the keyboard listener and rerun visualization
_, events = init_keyboard_listener()
init_rerun(session_name="recording")

# Connect the robot and teleoperator
robot.connect()
teleop.connect()

episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    print(f"########### Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        dataset=dataset,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

    # Reset the environment if not stopping or re-recording
    if (not events["stop_recording"]) and \
        (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1

# Clean up
log_say("Stop recording")
dataset.finalize()
robot.disconnect()
teleop.disconnect()
sleep(2)
# dataset.push_to_hub()

print(f"Dataset saved to: {dataset.repo_id}")
