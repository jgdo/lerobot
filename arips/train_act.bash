TIME=$(date +%F_%T)

lerobot-train --dataset.repo_id=jgdo/pick_and_place_1seq --policy.type=act  \
 --output_dir="outputs/train/act_pick_and_place_1seq_${TIME}"  \
  --job_name=act_pick_and_place_1seq  --policy.device=cuda \
  --wandb.enable=true --policy.push_to_hub=false --batch_size=32 \
  --num_workers=8 \
  --steps=30000 --eval_freq=1000 --save_freq=5000 \
  --dataset.image_transforms.enable=false 
 # --resume=true --config_path=outputs/train/act_pick_place_all/checkpoints/last/pretrained_model/train_config.json \

# lerobot-train --dataset.repo_id=jgdo/pick_and_place_3seq --policy.type=act  \
#  --output_dir="outputs/train/act_pick_and_place_3seq_${TIME}"  \
#   --job_name=act_pick_and_place_3seq  --policy.device=cuda \
#   --wandb.enable=true --policy.push_to_hub=false --batch_size=32 \
#   --num_workers=8 \
#   --steps=30000 --eval_freq=1000 --save_freq=5000 \
#   --dataset.image_transforms.enable=false 
