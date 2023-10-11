import os
import argparse
from pyhocon import ConfigFactory
import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.31.54', port=18888, stdoutToServer=True, stderrToServer=True)
from tools import axis_angle_to_matrix, matrix_to_rotation_6d
import torch
from models.builder import (
    build_pose_generator,
    build_motion_generator
)
from tqdm import tqdm
import numpy as np
import codecs as cs
from os.path import join as pjoin
from visualize import render_pose, render_motion


def main(conf_path):
    split_index = int(conf_path.split('/')[-1].split('_')[-1].split('.')[0])
    print(" finishing part " + str(split_index))
    print(" finishing part " + str(split_index))
    print(" finishing part " + str(split_index))
    with open(conf_path) as f:
        conf_text = f.read()
        f.close()
        conf = ConfigFactory.parse_string(conf_text)
    
    base_exp_dir = conf.get_string('general.base_exp_dir')
    mode = conf.get_string('general.mode')
    keyids = np.load(os.path.join(conf.get_string('general.text'), 'split_' + str(split_index) + '.npy'))


    for keyid in tqdm(keyids):
        with cs.open(pjoin('/mnt/disk_1/jinpeng/T2M/data/HumanML3D/texts', keyid + '.txt')) as f:
            text = f.readline().split('#')[0]
        pose_generator = build_pose_generator(dict(conf['pose_generator']))
        candidate_poses = pose_generator.get_topk_poses(text)
        if mode == 'pose':
            exit(0)
        motion_generator = build_motion_generator(dict(conf['motion_generator']))
        motion = motion_generator.get_motion(text, poses=candidate_poses)
        motion = axis_angle_to_matrix(motion[:, 3:].reshape(60, -1, 3))
        motion = matrix_to_rotation_6d(motion).reshape(60, -1).detach().cpu().numpy()
        padding = np.zeros((60, 3))
        motion = np.concatenate((padding, motion), axis=-1).T
        if not os.path.exists(os.path.join(base_exp_dir, keyid)):
            os.makedirs(os.path.join(base_exp_dir, keyid))
        npy_path = os.path.join(base_exp_dir, keyid, 'motion.npy')
        np.save(npy_path, motion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    main(args.conf)
