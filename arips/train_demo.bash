rm -rf outputs/train/train_demo

lerobot-train --dataset.repo_id=jgdo/pick_and_place_1seq --policy.type=act  \
 --output_dir=outputs/train/train_demo  \
  --job_name=train_demo  --policy.device=cuda \
  --wandb.enable=true --policy.push_to_hub=false --batch_size=32 \
  --num_workers=8 \
  --steps=30000 --eval_freq=1000 --save_freq=5000 \
  --dataset.image_transforms.enable=false 
