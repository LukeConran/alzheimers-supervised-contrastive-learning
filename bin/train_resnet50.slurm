#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=50:00:00
#SBATCH --mem=16G
#SBATCH --output=./output/train.%j
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

module load WebProxy/0000
python train.py \
    --train_json /scratch/group/optmai/chenlang/ADNI/text_data/train/data.json \
    --train_image_dir /scratch/group/optmai/chenlang/ADNI/image_data/train/ \
    --valid_json /scratch/group/optmai/chenlang/ADNI/text_data/validation/data.json \
    --valid_image_dir /scratch/group/optmai/chenlang/ADNI/image_data/validation/ \
    --lr 0.1 \
    --batch_size 16 \
    --epochs 80 \
    --backbone resnet \
    --model_name resnet50 \
    # --checkpoint_pretrain /scratch/user/baileyyeah/Alzheimer_my/checkpoints/resnet/resnet_50_23dataset.pth \