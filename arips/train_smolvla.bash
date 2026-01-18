OUT_DIR="outputs/train/smolvla_pick_place_train"

# rm -rf $OUT_DIR

lerobot-train \
  --dataset.repo_id=jgdo/pick_place_all_train \
  --policy.path=lerobot/smolvla_base  \
  --output_dir=$OUT_DIR \
  --job_name=smolvla_pick_place_train \
  --steps=50000 \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=false \
  --batch_size=60 \
  --policy.input_features='{"observation.images.top":  {"type":"VISUAL","shape":[3,256,256]}, "observation.images.front": {"type":"VISUAL","shape":[3,256,256]}}' \
  --policy.empty_cameras=1 \
  --num_workers=8 \
  --eval_freq=2000 --save_freq=10000 \
  --dataset.image_transforms.enable=false
 # --resume=true --config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
