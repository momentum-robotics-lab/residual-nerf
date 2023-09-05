import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='transforms.json')
parser.add_argument('--output', type=str, default='transforms.json')
args = parser.parse_args()

import json

with open(args.input, 'r') as f:
    transforms = json.load(f)

frames_old = transforms['frames']
frames_new = []
for frame in frames_old:
    path = frame['file_path']
    if './../Downloads/' in path:
        path = path.replace('./../Downloads/', '')
        frame['file_path'] = path
    frames_new.append(frame)

transforms['frames'] = frames_new

with open(args.output, 'w') as f:
    json.dump(transforms, f)
