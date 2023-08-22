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
export BG_CHECKPOINT='oats_bg/checkpoints/ngp_bg_ep0059.pth'
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/oats --workspace oats_normal -O --bound 2 --scale 0.33 --iters $ITERATIONS --type wrap  --bg_ckpt $BG_CHECKPOINT    \
--ckpt scratch \
--wandb --wandb_name oats_res --wandb_project oats 
