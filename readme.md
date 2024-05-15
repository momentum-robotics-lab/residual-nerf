# Residual-NeRF
### [Project Page](https://residual-nerf.github.io/) | [Video](https://youtu.be/oA2CkRFI61c?si=BLFGDBOXMaw_i4DD) | [Paper]() | [Data](https://drive.google.com/drive/folders/15r_mLt8MD5-AuYfoWCLy72hD27-tHA2N?usp=sharing)
Official implementation of Residual NeRF.<br><br>
[Residual-NeRF: Learning Residual NeRFs for Transparent Object Manipulation](https://residual-nerf.github.io/)  
 [Bardienus P. Duisterhof](https://bart-ai.com)<sup>1</sup>,
 [Yuemin Mao](https://yueminm.github.io/)<sup>1</sup>,
 [Si Heng Teng](https://www.ri.cmu.edu/ri-people/si-heng-teng/)<sup>1</sup>,
 [Jeffrey Ichnowski](https://ichnow.ski/)<sup>1</sup> <br>
 <sup>1</sup>Carnegie Mellon University, the Robotics Institute <br>
in International Conference on Robotics and Automation (ICRA) 2024

<img src='assets/method.png'/>

## Installation

### Docker
```bash
docker pull bartduis/torch-ngp:latest
docker run -it --runtime=nvidia --gpus all -v /path/to/Residual-NeRF:/workspace bartduis/torch-ngp:latest
```

### Install with pip
```bash
pip install -r requirements.txt
```

### Install with conda
```bash
conda env create -f environment.yml
conda activate torch-ngp
```

## Download the Data
You can access the data from [here](https://drive.google.com/drive/folders/15r_mLt8MD5-AuYfoWCLy72hD27-tHA2N?usp=sharing). You can also download all data with the following command:
```bash
gdown --folder https://drive.google.com/drive/folders/15r_mLt8MD5-AuYfoWCLy72hD27-tHA2N --remaining-ok 
```

The data should have the following structure:
```
data
├── bowl_rgb
│   ├── train 
│   │   ├── r_0_-1.png
│   │   ├── r_0_0.png
│   │   └── ...
│   ├── test
│   │   ├── r_0_-1.png
│   │   ├── r_0_0.png
│   │   └── ...
│   ├── val
│   │   ├── r_0_-1.png
│   │   ├── r_0_0.png
│   │   └── ...
│   └── gt_depth.npy
```
For each image filename, the first integer is the view id whereas the second integer indicates a background image (-1) or an image with the object present (0).

## Usage

The run_scripts folder contains the `generic_script_eval.sh` and `generic_script_train.sh` scripts. To train a vanilla, background, and residual NeRF on a dataset, run:

```bash 
./run_scripts/generic_script_train.sh <GPU_ID> <ITERATIONS> <DATA_PATH> <WORKSPACE> <DOWN_SCALE> <OUT_FOLDER> <N_VIEWS>
```
Here:
* <GPU_ID> Is the GPU to use for training.
* <ITERATION> Number iterations to train for.
* <DATA_PATH> Path to data folder in residual-NeRF format
* <WORKSPACE> Name of workspace output that'll be created.
* <DOWNSCALE> Downscale factor which can be used to train on smaller images, with the provided 800 x 800 dataset 1.0 makes most sense.
* <OUT_FOLDER> output directory.
* <N_VIEWS> Number of views used for training of the residual and vanilla nerf. This can be used to create an unbalanced dataset with fewer images of the full scene.

Here is an example of what training on the provided data could look like:
```bash
./run_scripts/generic_script_train.sh 0 10000 data/bowl bowl 1.0 output 100
```
We will provide better documentation of the evaluation script, and the scripts in the `scripts` folder in the coming weeks after initial release.

# Credits
This codebase built upon [Torch-NGP](https://github.com/ashawkey/torch-ngp), an excellent implementation of Instant-NGP in Pytorch with limited compromise. 

# Citation

    @inproceedings{duisterhof2024residualnerf,
      title={Residual-NeRF: Learning Residual NeRFs for Transparent Object Manipulation}, 
      author={Bardienus P. Duisterhof and Yuemin Mao and Si Heng Teng and Jeffrey Ichnowski},
      year={2024},
      booktitle={ICRA 2024}
    }

