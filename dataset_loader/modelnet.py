'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import random

import numpy as np
import warnings
import pickle
import torch

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc * pc, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud dataset, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, augmented=False, split='train'):
        self.root = root
        self.npoints = args.num_points
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.augmented = augmented

        if args.dataset == 'modelnet10':
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if args.dataset == 'modelnet10':
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        self.data_path = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt')
                              for i
                              in range(len(shape_ids[split]))]

    def __len__(self):
        return len(self.data_path)

    def _get_item(self, index):
        fn = self.data_path[index]
        cls = self.classes[self.data_path[index][0]]
        label = np.array([cls])
        point_set = np.loadtxt(fn[1], delimiter=',')

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.augmented:
            rotation = get_rand_rotation()
            point_set = np.matmul(point_set, rotation)

        point_set = pc_normalize(point_set)
        
        return point_set.astype(np.float32), label[0].astype(np.float32)

    def __getitem__(self, index):
        return self._get_item(index)


# rotation augmentation
def get_rand_rotation():
    alpha = random.uniform(0, 2*np.pi)
    beta = random.uniform(0, 2*np.pi)
    theta = random.uniform(0, 2*np.pi)
    rotation_x = np.array(
        [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    rotation_y = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    rotation_z = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.matmul(rotation_x, np.matmul(rotation_y, rotation_z))





