lerobot-train --dataset.repo_id=jgdo/pick_place_all_train --policy.type=act  \
 --output_dir=outputs/train/act_pick_place_all_train  \
  --job_name=act_pick_place_all_train  --policy.device=cuda \
  --wandb.enable=true --policy.push_to_hub=false --batch_size=32 \
  --num_workers=8 \
  --steps=100000 --eval_freq=5000 --save_freq=10000 \
  --dataset.image_transforms.enable=false \
  --policy.latent_dim=16 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=25
