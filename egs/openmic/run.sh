#!/bin/bash

set -x
export TORCH_HOME=../../pretrained_models

model=ast
dataset=openmic
imagenetpretrain=True
audiosetpretrain=True
bal=none
if [ $audiosetpretrain == True ]
then
  lr=1e-5
else
  lr=1e-4
fi
freqm=24
timem=96
mixup=0
epoch=30
batch_size=4
fstride=10
tstride=10
base_exp_dir=./exp/test-${dataset}-imp$imagenetpretrain-asp$audiosetpretrain-b$batch_size-lr${lr}_nor


echo 'now process fold'${fold}

exp_dir=${base_exp_dir}/
train_csv=/home/hc605/dataset/openmic-2018/train_music_full.csv
dev_csv=/home/hchen605/microphone_classification_add_feat/12class/data/dev_full.csv
test_csv=/home/hc605/dataset/openmic-2018/test_music_full.csv

  
CUDA_VISIBLE_DEVICES=1 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/run_music.py --model ${model} --dataset ${dataset} \
  --data-train ${train_csv} --data-val ${test_csv} --exp-dir $exp_dir \
  --label-csv ./data/esc_class_labels_indices.csv --n_class 20 \
  --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
  --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
  --tstride $tstride --fstride $fstride --imagenet_pretrain $imagenetpretrain --audioset_pretrain $audiosetpretrain
done

#python ./get_esc_result.py --exp_path ${base_exp_dir}