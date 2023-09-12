import argparse
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--csv',type=str,required=True)
args = parser.parse_args()

df = pd.read_csv(args.csv)


# get all the 'step' in numpy
steps = df['Step'].to_numpy()

# get all headers 
headers = df.columns.to_numpy()


# smooth psrn with time weighted EMA
psnr_res = df[headers[1]].ewm(span=1000).mean().to_numpy()
psnr_normal = df[headers[4]].ewm(span=1000).mean().to_numpy()

fontsize = 15
plt.plot(steps,psnr_res)
plt.plot(steps,psnr_normal)
plt.legend(['NeRF MiXeR','NeRF'],fontsize=fontsize)
plt.grid()
#set size of figure
plt.gcf().set_size_inches(10, 3)
plt.xlabel('Steps',fontsize=fontsize)
plt.ylabel('PSNR',fontsize=fontsize)
plt.title('PSNR vs Steps',fontsize=fontsize)
# set fontsize of ticks
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.savefig('psnr_vs_steps.pdf',bbox_inches='tight')
plt.show()