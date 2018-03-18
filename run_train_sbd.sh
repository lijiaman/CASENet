srun --gres=gpu:2 -c 8 -w dgx1 -p gpuc python main.py \
--batch-size=10 \
--multigpu \
2>&1|tee train_b4.log
