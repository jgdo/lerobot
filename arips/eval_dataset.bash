TRAIN_JOB="act_pick_and_place_1seq"
EVAL_DS="pick_and_place_1seq"
EVAL_OUT="outputs/eval_train/$TRAIN_JOB"

mkdir -p outputs/eval_train/
rm -rf $EVAL_OUT
cp -r outputs/train/$TRAIN_JOB outputs/eval_train/

lerobot-train --dataset.repo_id=jgdo/$EVAL_DS --policy.type=act  \
 --output_dir=$EVAL_OUT  \
  --job_name="eval_$TRAIN_JOB"  --policy.device=cuda \
  --wandb.enable=true --policy.push_to_hub=false --batch_size=32 \
  --num_workers=8 \
  --steps=2 --eval_freq=1 --save_freq=0 \
  --dataset.image_transforms.enable=false \
  --optimizer.lr=0.0 \
  --resume=true --config_path=$EVAL_OUT/checkpoints/last/pretrained_model/train_config.json 
