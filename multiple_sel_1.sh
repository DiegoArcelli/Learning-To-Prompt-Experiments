export CUDA_VISIBLE_DEVICES=2
python3.9 main.py         cifar100_l2p         --model vit_base_patch16_224         --batch-size 16         --data-path ./local_datasets/         --output_dir ./output_sel_1         --seed 729 > multiple_selection/output_selection_729
export CUDA_VISIBLE_DEVICES=2
python3.9 main.py         cifar100_l2p         --model vit_base_patch16_224         --batch-size 16         --data-path ./local_datasets/         --output_dir ./output_sel_1         --seed 615 > multiple_selection/output_selection_615
