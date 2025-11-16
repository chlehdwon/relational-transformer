dataset=$1
task=$2

cd ..
for seed in 1557 1601 4844; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run torchrun --standalone --nproc_per_node=4 scripts/pretrain_contd.py --seed $seed --dataset $dataset --task $task &
    CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run torchrun --standalone --nproc_per_node=4 scripts/pretrain_contd.py --seed $seed --dataset $dataset --task $task --use_fk &
    wait
    CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run torchrun --standalone --nproc_per_node=4 scripts/pretrain_contd.py --seed $seed --dataset $dataset --task $task --use_row &
    CUDA_VISIBLE_DEVICES=0,1,2,3 pixi run torchrun --standalone --nproc_per_node=4 scripts/pretrain_contd.py --seed $seed --dataset $dataset --task $task --use_fk --use_row &
    wait
done