#! /bin/bash
export OMP_NUM_THREADS=8
# if nothing provided set cuda visible device to 0 else to the first argument
if [ $# -eq 0 ]
  then
    echo "No arguments supplied, using GPU 0"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "Using GPU $1"
    export CUDA_VISIBLE_DEVICES=$1
fi

export ITERATIONS=10000
export BG_CHECKPOINT='lab_new_bg/checkpoints/ngp_bg_ep0100.pth'
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/lab_new --workspace lab_new_res -O --bound 16 --scale 1.0 --iters $ITERATIONS --type wrap  --bg_ckpt $BG_CHECKPOINT \
--ckpt scratch \
--wandb --wandb_name lab_res --wandb_project lab_new
