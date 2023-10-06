import argparse 
import os 
import shutil
import glob 
import math
from multiprocessing import Pool
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--data',type=str,required=True)
parser.add_argument('--min_views',type=int,default=5)
parser.add_argument('--max_views',type=int,default=100)
parser.add_argument('--step_views',type=int,default=15)
parser.add_argument('--out_dir',default='views_ablation')
parser.add_argument('--n_gpus',type=int,default=1)
args = parser.parse_args()


executable = 'python3 scripts/train_all_data.py '

def chunk_into_n(lst, n):
  size = math.ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )

def train_configs(data):
    process_idx = data[0]
    views = data[1]     
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(process_idx)

    for view in views:
        dir = os.path.join(args.out_dir,'views_'+str(view))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        command = executable + ' --gpu_id {} --data_location {} --downscale {} --out_folder {} --views {}'.format(process_idx,args.data,1.0,dir,view)

        process = subprocess.Popen(command,env=env,shell=True)
        process.wait()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
else:
    shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)


views = list(range(args.min_views,args.max_views+1,args.step_views))

views_chunked = chunk_into_n(views,args.n_gpus)
gpus = list(range(args.n_gpus))

# form gpu and data folder pairs
gpu_data_pairs = []
for i in range(args.n_gpus):
    gpu_data_pairs.append((gpus[i],views_chunked[i])) 

with Pool(args.n_gpus) as p:
    p.map(train_configs,gpu_data_pairs)