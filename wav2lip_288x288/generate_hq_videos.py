import os
import argparse
import shutil
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
args = parser.parse_args()

codeformer_cmd = 'cd ../CodeFormer && python inference_codeformer.py --bg_upsampler realesrgan --face_upsample -w 1.0 -s 1 --input_path {} --output_path {}'
base_dir = args.data_root  # '../LSR2/main'
sub_dirs = os.listdir(base_dir)

cmds = []

def execute_cmd(cmd):
    os.system(cmd[0])
    shutil.copy(cmd[1][0], cmd[1][1])
    os.system('rm -rf '+cmd[2])

for sub_dir in sub_dirs:
    sub_dir = os.path.join(base_dir, sub_dir)
    filenames = os.listdir(sub_dir)
    for filename in tqdm(filenames):
        if filename.endswith('mp4') and '_hq' not in filename:
            full_filename = os.path.join(sub_dir, filename)
            new_filename = full_filename[:-4]+'_hq'+full_filename[-4:]
            new_dirname = full_filename[:-4]
            # print(codeformer_cmd.format(full_filename, new_dirname))
            # print(os.path.join(new_dirname, filename), new_filename)
            if not os.path.exists(new_filename):
                cmds.append((codeformer_cmd.format(full_filename, new_dirname), (os.path.join(new_dirname, filename), new_filename), new_dirname))
                # os.system(codeformer_cmd.format(full_filename, new_dirname))
                # shutil.copy(os.path.join(new_dirname, filename), new_filename)

if len(cmds)>0:
    print('cmds:', len(cmds), cmds[0])
    with Pool(4) as p:
        p.map(execute_cmd, tqdm(cmds))
