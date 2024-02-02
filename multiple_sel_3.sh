export CUDA_VISIBLE_DEVICES=0
python3.9 main.py         cifar100_l2p         --model vit_base_patch16_224         --batch-size 16         --data-path ./local_datasets/         --output_dir ./output_sel_3         --seed 8 > multiple_selection/output_selection_8
