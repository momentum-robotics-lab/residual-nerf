import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--i', type=str, default='test.png', help='path to depth file')
parser.add_argument('--o', type=str, default='bg_output.png', help='path to depth file')
parser.add_argument('--x',type=int,default=1,help='x')
parser.add_argument('--y',type=int,default=1,help='y')
args = parser.parse_args()


# img = cv2.imread(args.i)
img = plt.imread(args.i)

stacked_img = np.tile(img,(args.x,args.y,1))

plt.imsave(args.o,stacked_img)
plt.imshow(stacked_img)
plt.show()

