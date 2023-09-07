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

export BG_CHECKPOINT='needles_bg/checkpoints/ngp_bg_ep0100.pth'
export ITERATIONS=10000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/needles --workspace needles_res -O --dt_gamma 0 --iters $ITERATIONS --type wrap --bg_ckpt $BG_CHECKPOINT  --mixnet_reg 10.0 --test --d_thresh 2.5 --mixnet_reg 10.0 --aabb_infer -0.6 0.12 -0.25 0.22 -0.54 0.28 


 
