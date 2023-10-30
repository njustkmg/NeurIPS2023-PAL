# Not All Out-of-Distribution Data Are Harmful to Open-Set Active Learning(NeurIPS 2023)

This is an PyTorch implementation of PAL.

## Requirement

* Python 3.7
* Pip packages:
  > pip install -r requirements.txt
  >

## Usage

### Dataset Preparation

This repository needs CIFAR10, CIFAR100 to train a model.

All datasets are supposed to be under ./data.

### Train

The basic usage of training involves utilizing 50 labeled data per class from the CIFAR-10 dataset: 

> python3 main.py --save_dir PAL_Cifar10_28/seed1 --out PAL_Cifar10_28/seed1 --miu 0.6 --max-query 10 --query-batch 1500 --dataset cifar10 --eval-step 70 --gpu-id 0 --need-ID 1450  --epochs 100 --num_classes 2 --num-labeled 50  --arch wideresnet --lambda_oem 0.1 --lambda_socr 0.5 --query-strategy PAL --batch-size 128 --lr 0.01 --expand-labels --seed 1  --mu 2
