# Ladder-Gamma-VAE
## Example usage for edge-prediction:
python train.py --dataset cora --hidden 64_32 --use_kl_warmup 1 --epochs 500 --cosine_norm 1 --reconstruct_x 1

## Example usage for semisupervised node classification:
python train_only.py --dataset citeseer --hidden 64_32 --use_kl_warmup 1 --epochs 100  --weight_decay 5e-3 --semisup_train 1 --dropout 0.5 --reconstruct_x 1
