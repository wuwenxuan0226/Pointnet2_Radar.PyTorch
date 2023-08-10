# Introduction

An unofficial implementation of [Semantic Segmentation on Radar Point Clouds](https://www.semanticscholar.org/paper/Semantic-Segmentation-on-Radar-Point-Clouds-Schumann-Hahn/fdf0969efe50d8b6d425f52c23062a5269c6a1a8).

## Requirements

- PyTorch, Python3, TensorboardX, tqdm, fire

## Radar Point Cloud Segmentation

- **Start**
- Dataset:
        [Radar Scenes](https://radar-scenes.com), download it from [Official Site](https://zenodo.org/record/4559821/files/RadarScenes.zip?download=1). Your data structure should be like this:

```
RadarScenes(dir)
        |- sequence_1(dir)
            |- camera(dir)
                |- 156859092964.jpg
                ...
            |- radar_data.h5
        |- sequence_2(dir)
            |- camera(dir)
            |- radar_data.h5
        ...
        |- sequence_158(dir)
        |- sensors.json
        |- sequences.json
```

- Train

```python
python train.py --data_root ./data/RadarScenes --npoints 3072 --nclasses 6 --log_dir seg_msg_radar --model pointnet2_seg_msg_radar --batch_size 4 --log_interval 1 --checkpoint_interval 2
```

- Evaluate

```python
python evaluate.py evaluate_seg pointnet2_seg_msg_radar ./data/RadarScenes seg_msg_radar/checkpoints/pointnet2_seg_24.pth batch_size npoints classes channels

eg.
python evaluate.py evaluate_seg pointnet2_seg_msg_radar ./data/RadarScenes seg_msg_radar/checkpoints/pointnet2_seg_24.pth 32 3072 6 5
```

- Visualize

```python
python plot_prediction.py plot_prediction_radarscenes pointnet2_seg_msg_radar ./data/RadarScenes seg_msg_radar/checkpoints/pointnet2_seg_24.pth index npoints classes

eg.
python plot_prediction.py plot_prediction_radarscenes pointnet2_seg_msg_radar ./data/RadarScenes seg_msg_radar/checkpoints/pointnet2_seg_24.pth 1 3072 6
```

- **Confusion Metrics**: [Figure](https://www.semanticscholar.org/paper/Semantic-Segmentation-on-Radar-Point-Clouds-Schumann-Hahn/fdf0969efe50d8b6d425f52c23062a5269c6a1a8/figure/4)
- Result in this project on validation set:
    |      |   Pred 1   |   Pred 2    |   Pred 3    |   Pred 4    |   Pred 5    |   Pred 6   |
    |:----:|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:----------:|
    | GT 1 |   82.4%    |    0.0%     |    0.2%     |    0.3%     |    4.0%     |   13.0%    |
    | GT 2 |    0.2%    |   51.9%     |   27.2%     |    0.5%     |    0.0%     |   20.1%    |
    | GT 3 |    0.2%    |    6.6%     |   81.6%     |    0.2%     |    0.0%     |   11.5%    |
    | GT 4 |    4.7%    |    0.9%     |    2.3%     |   81.6%     |    0.1%     |   10.5%    |
    | GT 5 |   13.3%    |    0.0%     |    0.1%     |    0.0%     |   72.0%     |   14.6%    |
    | GT 6 |    0.2%    |    0.0%     |    0.1%     |    0.0%     |    0.1%     |   99.6%    |

## Reference

- [https://github.com/charlesq34/pointnet2](https://github.com/charlesq34/pointnet2)
- [https://github.com/yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
- [https://github.com/zhulf0804/Pointnet2.PyTorch](https://github.com/zhulf0804/Pointnet2.PyTorch)
