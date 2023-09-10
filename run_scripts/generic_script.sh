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

export DATA_PATH=$3
export WORKSPACE=$4
export DOWNSCALE=$5

export BG_CHECKPOINT=$WORKSPACE'_bg/checkpoints/ngp_bg.pth'
export ITERATIONS=30000

# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi



# # run the normal version
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_normal' -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 2.0 --downscale $DOWNSCALE --min_near 0.2  \
--ckpt scratch --wandb --wandb_name $WORKSPACE'_normal' --wandb_project $WORKSPACE 

# run the background nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_bg' -O --dt_gamma 0  --iters $ITERATIONS --type bg --d_thresh 2.0 --downscale $DOWNSCALE --min_near 0.2  \
--ckpt scratch --wandb --wandb_name $WORKSPACE'_bg' --wandb_project $WORKSPACE 

# run the residual nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 2.0 --downscale $DOWNSCALE --min_near 0.2 --mixnet_reg 0.0001 \
--ckpt scratch --wandb --wandb_name $WORKSPACE'_res' --wandb_project $WORKSPACE  --bg_ckpt $BG_CHECKPOINT