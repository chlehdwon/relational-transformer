GPU=$1
dataset=$2

cd ..
for seed in 1557 1601 4844; do
    CUDA_VISIBLE_DEVICES=$GPU pixi run torchrun --standalone scripts/pretrain.py --seed $seed --dataset $dataset &
    CUDA_VISIBLE_DEVICES=$GPU pixi run torchrun --standalone scripts/pretrain_with_contrastive.py --seed $seed --dataset $dataset --use_fk &
    CUDA_VISIBLE_DEVICES=$GPU pixi run torchrun --standalone scripts/pretrain_with_contrastive.py --seed $seed --dataset $dataset --use_row &
    CUDA_VISIBLE_DEVICES=$GPU pixi run torchrun --standalone scripts/pretrain_with_contrastive.py --seed $seed --dataset $dataset --use_fk --use_row &
    wait
done