# nanoGPT's CPU recipe: a 4-layer GPT with 0.8M parameters, trained character by character on Shakespeare for 2,000 steps
git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
pip install torch numpy requests tiktoken
python data/shakespeare_char/prepare.py          # downloads about 1 MB of Shakespeare and numbers it by character
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char --device=cpu   # have it write a few passages of "Shakespeare"
