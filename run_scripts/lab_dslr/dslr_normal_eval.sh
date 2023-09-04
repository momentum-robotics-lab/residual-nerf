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
export ITERATIONS=10000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/lab_dslr --workspace lab_dslr_normal_downscale -O --scale 0.33 --bound 15.0  --iters $ITERATIONS --type wrap --d_thresh 2.0 --downscale 4.0 --test --min_near 0.7 --aabb_infer -1 1 -1 1 -1 1 --gui
