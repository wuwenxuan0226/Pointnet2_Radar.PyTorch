import torch
from utils.common import gather_points, get_dists


def ball_query(xyz, new_xyz, radius, K):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_indexes = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = get_dists(new_xyz, xyz)
    grouped_indexes[dists > radius] = N
    grouped_indexes = torch.sort(grouped_indexes, dim=-1)[0][:, :, :K]
    grouped_min_indexes = grouped_indexes[:, :, 0:1].repeat(1, 1, K)
    grouped_indexes[grouped_indexes == N] = grouped_min_indexes[grouped_indexes == N]
    return grouped_indexes