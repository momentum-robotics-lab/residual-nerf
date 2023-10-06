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
export N_VIEWS=$7

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


# echo "Downscale: $DOWNSCALE"
# # run the normal version
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_normal' -O --dt_gamma 0 --scale 2.0 --bound 4   --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101  \
--results_dir $OUT_FOLDER --ckpt scratch  --n_imgs $N_VIEWS
# --wandb --wandb_name $WORKSPACE'_normal' --wandb_project $WORKSPACE --n_imgs $N_VIEWS 
# # --max_training_time 300

# # # # # # run the background nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_bg' -O --dt_gamma 0  --scale 2.0 --bound 4  --iters $ITERATIONS --type bg --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101 \
--ckpt scratch --results_dir $OUT_FOLDER 
# --wandb --wandb_name $WORKSPACE'_bg' --wandb_project $WORKSPACE
# # # # run the residual nerf

python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' --bg_ckpt $BG_CHECKPOINT --scale 2.0 --bound 4  -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --mixnet_reg 0.00 --eval_interval 101 \
--results_dir $OUT_FOLDER --ckpt scratch --n_imgs $N_VIEWS 
# --wandb --wandb_name $WORKSPACE'_res' --wandb_project $WORKSPACE 
# --max_training_time 300

## LEGO
# echo "Downscale: $DOWNSCALE"
# # # # run the normal version
# # python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_normal' -O --dt_gamma 0 --scale 0.8 --bound 1   --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101  \
# # --results_dir $OUT_FOLDER --ckpt scratch  --wandb --wandb_name $WORKSPACE'_normal' --wandb_project $WORKSPACE
# # # # --max_training_time 300

# # # # # # # # run the background nerf
# # python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_bg' -O --dt_gamma 0  --scale 0.8 --bound 1  --iters $ITERATIONS --type bg --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 101 \
# # --ckpt scratch --wandb --wandb_name $WORKSPACE'_bg' --wandb_project $WORKSPACE --results_dir $OUT_FOLDER

# # # # # run the residual nerf

# python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' --bg_ckpt $BG_CHECKPOINT --scale 0.8 --bound 1  -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --mixnet_reg 0.00 --eval_interval 101 \
# --results_dir $OUT_FOLDER --ckpt scratch --wandb --wandb_name $WORKSPACE'_res' --wandb_project $WORKSPACE

