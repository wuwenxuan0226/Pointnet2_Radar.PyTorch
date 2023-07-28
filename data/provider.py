'''
author: charlesq34
addr: https://github.com/charlesq34/pointnet2/blob/master/utils/provider.py

update: zhulf
'''

import numpy as np


def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc


def convert_to1(data):
    mean = np.mean(data, axis=0)
    data -= mean
    m = np.max(np.abs(data))
    data /= m
    return data


def shuffle_points(pc):
    idx = np.arange(pc.shape[0])
    np.random.shuffle(idx)
    return pc[idx,:]


def rotate_point_cloud(pc):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [-sinval, 0, cosval],
                                [0, 1, 0]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (1, C))
    pc += shifts
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc


def augment_pc(pc):
    rotated_pc = rotate_point_cloud(pc[:, :3])
    jittered_pc = shift_point_cloud(rotated_pc)
    jittered_pc = jitter_point_cloud(jittered_pc)
    pc[:, :3] = jittered_pc
    return pc


if __name__ == '__main__':
    pc = np.random.randn(4, 6)
    print(pc)
    pc = augment_pc(pc)
    print(pc)