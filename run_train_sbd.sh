srun --gres=gpu:1 -c 4 -w dgx1 -p gpuc python main.py \
--batch-size=1 \
--workers=1 \
2>&1|tee train_b4.log
