# Continuing from the previous step, inside the nanoGPT folder: the same character-level pipeline, now with Tang poems
python make_tang.py
cp data/shakespeare_char/prepare.py data/tang_char/   # input.txt is already there, so the script won't download Shakespeare again
python data/tang_char/prepare.py
python train.py config/train_shakespeare_char.py --dataset=tang_char --out_dir=out-tang --device=cpu --compile=False --eval_iters=20 --log_interval=100 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-tang --device=cpu --num_samples=3 --max_new_tokens=60
