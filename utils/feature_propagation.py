import torch
import torch.nn as nn
from utils.common import gather_points, get_dists


def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), indexes: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, indexes = torch.sort(dists, dim=-1)
    dists, indexes = dists[:, :, :3], indexes[:, :, :3]
    return dists, indexes

def two_nn(xy1, xy2):
    '''

    :param xy1: shape=(B, N1, 2)
    :param xy2: shape=(B, N2, 2)
    :return: dists: shape=(B, N1, 2), indexes: shape=(B, N1, 2)
    '''
    dists = get_dists(xy1, xy2)
    dists, indexes = torch.sort(dists, dim=-1)
    dists, indexes = dists[:, :, :2], indexes[:, :, :2]
    return dists, indexes


def three_interpolate(xyz1, xyz2, points2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = points2.shape
    dists, indexes = three_nn(xyz1, xyz2)
    inverse_dists = 1.0 / (dists + 1e-8)
    weight = inverse_dists / torch.sum(inverse_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
    interpolated_points = gather_points(points2, indexes)  # shape=(B, N1, 3, C2)
    interpolated_points = torch.sum(weight * interpolated_points, dim=2)
    return interpolated_points


class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channels, mlp, bn=True):
        super(PointNet_FP_Module, self).__init__()
        self.backbone = nn.Sequential()
        bias = False if bn else True
        for i, out_channels in enumerate(mlp):
            self.backbone.add_module('Conv_{}'.format(i), nn.Conv2d(in_channels,
                                                                    out_channels,
                                                                    1,
                                                                    stride=1,
                                                                    padding=0,
                                                                    bias=bias))
            if bn:
                self.backbone.add_module('Bn_{}'.format(i), nn.BatchNorm2d(out_channels))
            self.backbone.add_module('Relu_{}'.format(i), nn.ReLU())
            in_channels = out_channels
    def forward(self, xyz1, xyz2, points1, points2):
        '''

        :param xyz1: shape=(B, N1, 3)
        :param xyz2: shape=(B, N2, 3)   (N1 >= N2)
        :param points1: shape=(B, N1, C1)
        :param points2: shape=(B, N2, C2)
        :return: new_points2: shape = (B, N1, mlp[-1])
        '''
        B, N1, C1 = points1.shape
        _, N2, C2 = points2.shape
        if N2 == 1:
            interpolated_points = points2.repeat(1, N1, 1)
        else:
            interpolated_points = three_interpolate(xyz1, xyz2, points2)
        cat_interpolated_points = torch.cat([interpolated_points, points1], dim=-1).permute(0, 2, 1).contiguous()
        new_points = torch.squeeze(self.backbone(torch.unsqueeze(cat_interpolated_points, -1)), dim=-1)
        return new_points.permute(0, 2, 1).contiguous()