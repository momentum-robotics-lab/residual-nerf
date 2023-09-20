import numpy as np
import json 
import os  
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process colmap poses for easy robot execution.')
    parser.add_argument('--pose', type=str, help='path to pose json file.',default='poses_robot.json')
    parser.add_argument('--dex_output' , type=str, help='Path to dex output json file.')
    args = parser.parse_args()

