import argparse
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import glob 
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--folder',type=str,required=True)
parser.add_argument('--psnr_min',type=float,default=20.0)
parser.add_argument('--psnr_max',type=float,default=35.0)
parser.add_argument('--debug',action='store_true')
args = parser.parse_args()

csvs = glob.glob(os.path.join(args.folder,'*.csv'))

for csv in csvs:

    df = pd.read_csv(csv)

    def format_ticks(x, pos):
        if x >= 1000:
            return '{:.0f}k'.format(x*1e-3)
        return '{:.0f}'.format(x)

    # get all the 'step' in numpy
    steps = df['Step'].to_numpy()

    # get all headers 
    headers = df.columns.to_numpy()


    # smooth psrn with time weighted EMA
    psnr_res = df[headers[1]].to_numpy()
    psnr_normal = df[headers[4]].to_numpy()

    # remove last entry (not val psnr)
    psnr_res = psnr_res[:-1]
    psnr_normal = psnr_normal[:-1]
    steps = steps[:-1]

    fontsize = 15
    plt.plot(steps,psnr_res)
    plt.plot(steps,psnr_normal)
    plt.legend(['NeRF MiXeR','NeRF'],fontsize=fontsize,loc='lower right')
    plt.grid()
    #set size of figure
    plt.gcf().set_size_inches(10, 2)
    plt.xlabel('Steps',fontsize=fontsize)
    plt.ylabel('PSNR',fontsize=fontsize)
    # set fontsize of ticks
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(args.psnr_min,args.psnr_max)

    # express steps in thousands, e.g. 1000 -> 1k
    plt.ticklabel_format(style='plain',axis='x',useOffset=False)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_ticks))

    name = csv[:-4]+'.pdf'

    plt.savefig(name,bbox_inches='tight')
    if args.debug:
        plt.show()
    plt.close()