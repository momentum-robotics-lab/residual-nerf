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

export BG_CHECKPOINT='lab_dslr_bg/checkpoints/ngp_bg_ep0371.pth'
export ITERATIONS=30000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/lab_dslr --workspace lab_dslr_res_bgsigma -O --dt_gamma 0 --scale 0.33 --bound 15.0 --iters $ITERATIONS --bg_ckpt $BG_CHECKPOINT --type wrap --d_thresh 3.5 --downscale 4.0 --min_near 0.1 \
--ckpt scratch --wandb --wandb_name lab_dslr_res_4x_downscale_bgsigma --wandb_project lab_dslr 

 
