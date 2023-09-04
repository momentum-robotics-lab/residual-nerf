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

export BG_CHECKPOINT='vase_bg/checkpoints/ngp_bg.pth'
export ITERATIONS=30000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/dslr_cup --workspace dslr_cup_normal -O --scale 0.33 --bound 15.0  --iters $ITERATIONS --type wrap --d_thresh 3.5 --downscale 4.0 --min_near 0.5  \
--ckpt scratch --wandb --wandb_name dslr_cup_normal --wandb_project dslr_cup 

