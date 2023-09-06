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

export BG_CHECKPOINT='dslr_wine_bg/checkpoints/ngp_bg_ep0323.pth'
export ITERATIONS=30000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi


python3 main_nerf.py data/dslr_wine --workspace dslr_wine_res -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 2.5 --bg_ckpt $BG_CHECKPOINT --downscale 4.0 --min_near 0.2 --test --aabb_infer -4.0 0.33 -1.10 0.77  -0.66 0.66
