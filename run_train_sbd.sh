srun --gres=gpu:1 -c 4 -w dgx1 -p gpuc python main.py \
--lr=0.01 \
--batch-size=1 \
--workers=1 \
--epochs=300000 \
2>&1|tee train_b4.log
