srun --gres=gpu:1 -c 4 -w guppy36 -p gpuc python main.py \
--lr=0.1 \
--batch-size=1 \
--workers=8 \
--epochs=300000 \
--lr-steps 2500 5000 7500 10000 \
2>&1|tee train_b4_lr0.1.log
