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
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/clutter --workspace clutter_normal -O --bound 1 --scale 2.0 --dt_gamma 0 --iters $ITERATIONS --type wrap  --ckpt scratch --wandb --wandb_name clutter_normal_large_density_grid_low_density_thresh --wandb_project clutter --density_thresh 0.01 
