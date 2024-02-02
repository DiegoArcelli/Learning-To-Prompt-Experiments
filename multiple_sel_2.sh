export CUDA_VISIBLE_DEVICES=3
python3.9 main.py         cifar100_l2p         --model vit_base_patch16_224         --batch-size 16         --data-path ./local_datasets/         --output_dir ./output_sel_2         --seed 105 > multiple_selection/output_selection_105
export CUDA_VISIBLE_DEVICES=3
python3.9 main.py         cifar100_l2p         --model vit_base_patch16_224         --batch-size 16         --data-path ./local_datasets/         --output_dir ./output_sel_2         --seed 97 > multiple_selection/output_selection_97
