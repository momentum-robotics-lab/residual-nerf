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
# run the normal version
python3 main_nerf.py $DATA_PATH --bound 4 --scale 2.0 --workspace $WORKSPACE'_normal' -O --dt_gamma 0   --iters $ITERATIONS --type wrap --d_thresh 2.5 --downscale $DOWNSCALE --min_near 0.2 --eval_interval 10  \
--results_dir $OUT_FOLDER --test --aabb_infer -0.3 0.3 -0.3 0.3 -0.3 0.3
# --gui 
# --aabb_infer  -0.9 0.4 -0.7 2.3 -1 1.2 --scale 2 --bound 4
# --gui --aabb_infer  -0.9 0.4 -0.7 2.3 -1 1.2
# 


# # # run the residual nerf
# python3 main_nerf.py $DATA_PATH --workspace $WORKSPACE'_res' --bg_ckpt $BG_CHECKPOINT -O --dt_gamma 0  --iters $ITERATIONS --type wrap --d_thresh 3.0 --downscale $DOWNSCALE --min_near 0.2 --mixnet_reg 0.00 --eval_interval 10 \
# --results_dir $OUT_FOLDER --test --gui --aabb_infer  -0.9 0.4 -0.7 2.3 -1 1.2 --scale 2 --bound 4
# --aabb_infer -0.3 0.3 -0.3 0.3 -0.3 0.3
# --gui --aabb_infer  -0.9 0.4 -0.7 2.3 -1 1.2
# --gui --aabb_infer 0.0 1.4  0.0 1.4 0.0 0.6
# # # --aabb_infer