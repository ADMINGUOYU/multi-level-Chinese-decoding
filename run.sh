#!/bin/bash
## Du-IN
cd ./train/duin
python run_align_semantic.py --seeds 42 --subjs 001 --subj_idxs 0 \
     --pt_ckpt ./pretrains/duin/001/mae/model/checkpoint-399.pth \