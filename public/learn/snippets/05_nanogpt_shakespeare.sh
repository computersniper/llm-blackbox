# nanoGPT 的 CPU 配方：4 层、80 万参数的小 GPT，在莎士比亚全集上逐字符训练 2000 步
git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
pip install torch numpy requests tiktoken
python data/shakespeare_char/prepare.py          # 下载约 1 MB 的莎士比亚文本，按字符编号
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char --device=cpu   # 让它写几段“莎士比亚”
