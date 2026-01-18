# lerobot-eval \
#     --policy.path=/home/jgdo/dev/lerobot/outputs/train/act_so101_test \
#     --eval.batch_size=10 \
#     --eval.n_episodes=10 \
#     --policy.use_amp=false \
#     --policy.device=cuda

rm -rf /home/jgdo/.cache/huggingface/lerobot/jgdo/eval_act

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyUSB0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30}}" \
  --robot.id=my_lerobot \
  --display_data=true \
  --dataset.repo_id=jgdo/eval_act \
  --dataset.single_task="Pick and place object" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=500 \
  --policy.path=/home/jgdo/dev/lerobot/outputs/act/jgdo/pick_and_place_3seq_2026-01-17_23:20:07/checkpoints/last/pretrained_model
