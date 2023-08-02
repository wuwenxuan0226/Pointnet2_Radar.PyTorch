import matplotlib.pyplot as plt
import torch
import fire
from models.pointnet2_seg import pointnet2_seg_msg_radar, seg_loss
from data.RadarScenes import RadarScenes


def plot_prediction_radarscenes(model_name, data_root, checkpoint, index, npoints, nclasses, dims=5):
    # this function is used to select a frame from the radar data and plot the original point cloud and the prediction
    Models = {
        'pointnet2_seg_msg_radar': pointnet2_seg_msg_radar
    }
    Model = Models[model_name]
    dataset_test = RadarScenes(data_root=data_root, split='validation', npoints=npoints, combined_frame_num=28, augment=False, dp=False, normalize=False)
    device = torch.device('cuda')
    model = Model(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Index: {}, Plotting..".format(index))
    predictions, labels = [], []
    loss_func = seg_loss().to(device)
    data, label = dataset_test.__getitem__(index)
    # numpy.ndarray to torch.tensor
    data = torch.from_numpy(data).unsqueeze(0)
    label = torch.from_numpy(label).unsqueeze(0)
    labels.append(label)
    xyz, points = data[:, :, :3], data[:, :, 3:]

    with torch.no_grad():
        prediction = model(xyz.to(device), points.to(device))
        prediction = torch.max(prediction, dim=1)[1].cpu().detach().numpy()
        predictions.append(prediction)

    # show index position
    file_name, timestamp, frame, total_frame = dataset_test.get_index_position(index)
    print('File name: {}, Timestamp: {}, Frame: {}/{}'.format(file_name, timestamp, frame, total_frame - 1))

    # separate the xyz according to the labels
    xyz_label_0 = xyz.numpy()[0, label.numpy()[0] == 0, :]
    xyz_label_1 = xyz.numpy()[0, label.numpy()[0] == 1, :]
    xyz_label_2 = xyz.numpy()[0, label.numpy()[0] == 2, :]
    xyz_label_3 = xyz.numpy()[0, label.numpy()[0] == 3, :]
    xyz_label_4 = xyz.numpy()[0, label.numpy()[0] == 4, :]
    xyz_label_5 = xyz.numpy()[0, label.numpy()[0] == 5, :]

    # separate the xyz according to the predictions
    xyz_pred_0 = xyz.numpy()[0, prediction[0] == 0, :]
    xyz_pred_1 = xyz.numpy()[0, prediction[0] == 1, :]
    xyz_pred_2 = xyz.numpy()[0, prediction[0] == 2, :]
    xyz_pred_3 = xyz.numpy()[0, prediction[0] == 3, :]
    xyz_pred_4 = xyz.numpy()[0, prediction[0] == 4, :]
    xyz_pred_5 = xyz.numpy()[0, prediction[0] == 5, :]

    # plot the original point cloud and prediction in subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    if xyz_label_0.size != 0:
        ax1.scatter(xyz_label_0[:, 1], xyz_label_0[:, 0], c='red', s=5, label='CAR')
    if xyz_label_1.size != 0:
        ax1.scatter(xyz_label_1[:, 1], xyz_label_1[:, 0], c='orange', s=5, label='PEDESTRIAN')
    if xyz_label_2.size != 0:
        ax1.scatter(xyz_label_2[:, 1], xyz_label_2[:, 0], c='purple', s=5, label='PEDESTRIAN_GROUP')
    if xyz_label_3.size != 0:
        ax1.scatter(xyz_label_3[:, 1], xyz_label_3[:, 0], c='green', s=5, label='TWO_WHEELER')
    if xyz_label_4.size != 0:
        ax1.scatter(xyz_label_4[:, 1], xyz_label_4[:, 0], c='blue', s=5, label='LARGE_VEHICLE')
    if xyz_label_5.size != 0:
        ax1.scatter(xyz_label_5[:, 1], xyz_label_5[:, 0], c='grey', s=1, label='STATIC')
    ax1.set_title('Ground Truth')
    ax1.legend()
    # reverse the x axis
    ax1.set_xlim(ax1.get_xlim()[::-1])
    # add axis name
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')

    if xyz_pred_0.size != 0:
        ax2.scatter(xyz_pred_0[:, 1], xyz_pred_0[:, 0], c='red', s=5, label='CAR')
    if xyz_pred_1.size != 0:
        ax2.scatter(xyz_pred_1[:, 1], xyz_pred_1[:, 0], c='orange', s=5, label='PEDESTRIAN')
    if xyz_pred_2.size != 0:
        ax2.scatter(xyz_pred_2[:, 1], xyz_pred_2[:, 0], c='purple', s=5, label='PEDESTRIAN_GROUP')
    if xyz_pred_3.size != 0:
        ax2.scatter(xyz_pred_3[:, 1], xyz_pred_3[:, 0], c='green', s=5, label='TWO_WHEELER')
    if xyz_pred_4.size != 0:
        ax2.scatter(xyz_pred_4[:, 1], xyz_pred_4[:, 0], c='blue', s=5, label='LARGE_VEHICLE')
    if xyz_pred_5.size != 0:
        ax2.scatter(xyz_pred_5[:, 1], xyz_pred_5[:, 0], c='grey', s=1, label='STATIC')
    ax2.set_title('Prediction')
    ax2.legend()
    # reverse the x axis
    ax2.set_xlim(ax2.get_xlim()[::-1])
    # add axis name
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')

    plt.show()


if __name__ == '__main__':
    fire.Fire()