srun --gres=gpu:2 -c 8 -w dgx1 -p gpuc python main.py \
--multigpu \
--lr=1e-7 \
--batch-size=10 \
--workers=1 \
--epochs=40000 \
--lr-steps 10000 20000 30000 \
--pretrained-model="./official_models/Init_CASENet.pth.tar" \
2>&1|tee train_b10_lr1e-7.log
