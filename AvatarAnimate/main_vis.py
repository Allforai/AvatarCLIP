import os
import argparse
from pyhocon import ConfigFactory
import pydevd_pycharm
# pydevd_pycharm.settrace('10.81.48.3', port=18888, stdoutToServer=True, stderrToServer=True)
from tools import axis_angle_to_matrix, matrix_to_rotation_6d, axis_angle_to
import torch
from models.builder import (
    build_pose_generator,
    build_motion_generator
)
from tqdm import tqdm
import numpy as np
import codecs as cs
from os.path import join as pjoin
from ood_dataset import OOD_Dataset
from visualize import render_pose, render_motion
# import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.31.54', port=18888, stdoutToServer=True, stderrToServer=True)

def main(conf_path):
    # print("conf_path:", conf_path)
    split_index = int(conf_path.split('/')[-1].split('_')[-1].split('.')[0])
    with open(conf_path) as f:
        conf_text = f.read()
        f.close()
        conf = ConfigFactory.parse_string(conf_text)

    base_exp_dir = '/mnt/disk_1/jinpeng/T2M/teaser/Avatarclip/'
    mode = conf.get_string('general.mode')
    text = 'Holding head and crying bitterly, and finally crouched down'
    keyid = text.split(' ')[-1]
    if not os.path.exists(os.path.join(base_exp_dir, keyid)):
        os.makedirs(os.path.join(base_exp_dir, keyid))
    print(os.path.join(base_exp_dir, keyid))
    pose_generator = build_pose_generator(dict(conf['pose_generator']))
    candidate_poses = pose_generator.get_topk_poses(text)
    if mode == 'pose':
        exit(0)
    motion_generator = build_motion_generator(dict(conf['motion_generator']))
    motion = motion_generator.get_motion(text, poses=candidate_poses)
    motion = torch.cat((torch.tensor([np.pi / 4, -np.pi / 2, -np.pi / 2]).to('cuda').repeat(60, 1), motion[:, :63]), dim=-1).reshape(60, 22, -1)
    motion = axis_angle_to_matrix(motion)
    motion = matrix_to_rotation_6d(motion).reshape(60, -1).detach().cpu().numpy()
    padding = np.zeros((60, 3))
    motion = np.concatenate((padding, motion), axis=-1)[None]
    npy_path = os.path.join(base_exp_dir, keyid, 'motion.npy')
    np.save(npy_path, motion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args.conf)
