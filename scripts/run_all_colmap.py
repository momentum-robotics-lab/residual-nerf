import glob 
import os

dataset_dir = 'real_dataset' 
results_dir = 'colmap_results'

if os.path.exists(results_dir):
    print('Results directory already exists')
    # ask if want to overwrite
    overwrite = input('Overwrite? (y/n): ')
    if overwrite != 'y':
        exit()
else:
    os.mkdir(results_dir)

# Get all the folders in the dataset directory
folders = glob.glob(os.path.join(dataset_dir, '**'))

# filter out the non-folders
folders = [f for f in folders if os.path.isdir(f) and 'background' not in f]
background_folder = os.path.join(dataset_dir, 'background')
# Run colmap on all the folders
for folder in folders:
    # Get the folder name
    name = os.path.basename(os.path.normpath(folder))
    scene_dir = os.path.join(results_dir, name)
    if os.path.exists(scene_dir):
        print('Scene directory already exists')
        os.system('rm -rf {}'.format(scene_dir))

    os.mkdir(scene_dir)

    image_dir = os.path.join(scene_dir, 'images')
    os.mkdir(image_dir)
    json_file = 'transforms.json'

    # Run colmap on the folder
    os.system('python3 scripts/colmap_prep.py --images_bg {} --images_wrap {} --images {} --out {}'.format(background_folder,folder,image_dir,json_file))

