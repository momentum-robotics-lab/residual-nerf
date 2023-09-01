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

export BG_CHECKPOINT='wine_bg/checkpoints/ngp_bg_ep0100.pth'
export ITERATIONS=10000
# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi

python3 main_nerf.py data/wine --workspace wine_res -O --bound 4 --scale 0.8 --dt_gamma 0 --iters $ITERATIONS --type wrap --bg_ckpt $BG_CHECKPOINT --mixnet_reg 10.0 \
--ckpt scratch --wandb --wandb_name wine_res_switch_bg --wandb_project wine 

 
