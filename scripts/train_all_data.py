import glob 
import os 
import subprocess 
import time 
import argparse
import os
import subprocess
from multiprocessing import Pool
import shutil
import math

parser = argparse.ArgumentParser()
parser.add_argument('--n_gpus',type=int,default=1)
parser.add_argument('--data_location',type=str,default='data')
args = parser.parse_args()

def chunk_into_n(lst, n):
  size = math.ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )
  
def run_batch(gpu_data_pairs):
    process_idx = gpu_data_pairs[0]
    data_folders = gpu_data_pairs[1]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(process_idx)
    for data_folder in data_folders:
        print('Running on {}'.format(data_folder))
        name = os.path.basename(os.path.normpath(data_folder))
        process = subprocess.Popen('./run_scripts/generic_script.sh 0 10000 {} {}'.format(data_folder,name),env=env,shell=True)
        process.wait()


data_location = args.data_location
data_folders = glob.glob(os.path.join(data_location, '**'))
# filter out non-dirs
data_folders = [f for f in data_folders if os.path.isdir(f)]

data_folders_chunked = chunk_into_n(data_folders,args.n_gpus)
gpus = list(range(args.n_gpus))

# form gpu and data folder pairs
gpu_data_pairs = []
for i in range(args.n_gpus):
    gpu_data_pairs.append((gpus[i],data_folders_chunked[i]))    

processes = []



with Pool(args.n_gpus) as p:
    p.map(run_batch,gpu_data_pairs)



