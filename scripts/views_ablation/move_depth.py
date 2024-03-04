import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='depth', help='source folder')
parser.add_argument('--dst', type=str, default='depth', help='destination folder')
parser.add_argument('--clean_dst',action='store_true', help='clean destination folder')
args = parser.parse_args()


# find all .npz files in the source folder structure
import os

scenes = ['bowl','drink_flat','drink_up','wine_down']

# clean up dst
if args.clean_dst:
    # find all folders named dex
    for root, dirs, files in os.walk(args.dst):
        for dir in dirs:
            if 'dex' in dir or 'ours' in dir :
                global_dir = os.path.join(root, dir)
                
                # remove directory 
                os.system('rm -rf ' + global_dir)
                
                #make directory
                os.makedirs(global_dir)
    
            



def find_npz_files(data_dir):
    npz_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                full_path = os.path.join(root, file)
                if 'val' in full_path:
                    npz_files.append(full_path)
    return npz_files


def analyze_filename(filename):
    # print('filename: ', filename)
    
    file_parts = filename.split('/')
    views = file_parts[1]
    #extract integer
    views = views.split('_')[1]
    
    scene_type_full = file_parts[2]
    
    for scene in scenes:
        if scene in scene_type_full:
            scene_type = scene
            break
    
    method = None
    
    if scene_type_full.split('_')[-1] == 'res':
        method = 'ours'
    elif scene_type_full.split('_')[-1] == 'normal':
        method = 'dex'
    
    return views, scene_type, method
    
    




npz_files = find_npz_files(args.src)
print(npz_files)
for npz_file in npz_files:
    views, scene_type, method = analyze_filename(npz_file)
    if method is not None:
        goal_file = os.path.join(args.dst, scene_type,method,'depth_'+views+'.npz')
        print('copying from ', npz_file, ' to ', goal_file)
        os.system('cp '+npz_file+' '+goal_file)