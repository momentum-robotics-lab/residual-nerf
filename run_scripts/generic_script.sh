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
export OUT_FOLDER=$6

export BG_CHECKPOINT=$OUT_FOLDER'/'$WORKSPACE'_bg/checkpoints/'
export ITERATIONS=30000

# second argument is number of iterations 
if [ $# -lt 2 ]
  then
    echo "No iterations supplied, using 10000"
    
else
    echo "Using $2 iterations"
    export ITERATIONS=$2
fi


echo "Downscale: $DOWNSCALE"
# # run the normal version
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_normal' -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101  \
--results_dir $OUT_FOLDER --ckpt scratch  --wandb --wandb_name $WORKSPACE'_normal' --wandb_project $WORKSPACE --n_imgs 10 

# # # # run the background nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_bg' -O --dt_gamma 0  --iters $ITERATIONS --type bg --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101 \
--ckpt scratch --wandb --wandb_name $WORKSPACE'_bg' --wandb_project $WORKSPACE --results_dir $OUT_FOLDER

# # run the residual nerf

python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' --bg_ckpt $BG_CHECKPOINT -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --mixnet_reg 0.00 --eval_interval 100 \
--results_dir $OUT_FOLDER --ckpt scratch --wandb --wandb_name $WORKSPACE'_res' --wandb_project $WORKSPACE --n_imgs 10