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
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_normal' -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 2.5 --downscale 4.0 --min_near 0.2  \
--test --aabb_infer -4.0 0.33 -1.10 0.77  -0.66 0.66 

# run the background nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_bg' -O --dt_gamma 0  --iters $ITERATIONS --type bg --d_thresh 2.5 --downscale 4.0 --min_near 0.2  \
--test --aabb_infer -4.0 0.33 -1.10 0.77  -0.66 0.66 
# run the residual nerf
python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 2.5 --downscale 4.0 --min_near 0.2 --mixnet_reg 0.001 \
 --bg_ckpt $BG_CHECKPOINT --test --aabb_infer -4.0 0.33 -1.10 0.77  -0.66 0.66 