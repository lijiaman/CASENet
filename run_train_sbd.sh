srun --gres=gpu:2 -c 8 -w guppy34 -p gpuc python main.py \
--multigpu \
--lr=1e-7 \
--batch-size=2 \
--workers=1 \
--epochs=40000 \
--lr-steps 10000 20000 30000 \
2>&1|tee train_b10_lr1e-7.log
