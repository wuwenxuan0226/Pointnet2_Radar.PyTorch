import fire
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.pointnet2_seg import pointnet2_seg_msg_radar, seg_loss
from data.RadarScenes import RadarScenes
from utils.IoU import compute_metrics


def evaluate_seg(model_name, data_root, checkpoint, batch_size, npoints, nclasses, dims):
    print('Loading..')
    Models = {
        'pointnet2_seg_msg_radar': pointnet2_seg_msg_radar
    }
    Model = Models[model_name]
    if model_name == 'pointnet2_seg_msg_radar':
        dataset_test = RadarScenes(data_root=data_root, split='validation', npoints=npoints, combined_frame_num=28)
    
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    device = torch.device('cuda')
    model = Model(dims, nclasses)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print('Loading {} completed'.format(checkpoint))
    print("Dataset: {}, Evaluating..".format(len(dataset_test)))
    loss_func = seg_loss().to(device)
    
    losses, predictions, labels = [], [], []
    for data, label in tqdm(test_loader):
        label = label.long().to(device)
        xyz, points = data[:, :, :3], data[:, :, 3:]
        with torch.no_grad():
            prediction = model(xyz.to(device), points.to(device))
            loss = loss_func(prediction, label)
            prediction = torch.max(prediction, dim=1)[1].cpu().detach().numpy()
            predictions.append(prediction)
            losses.append(loss.item())
            labels.append(label.cpu())
    if model_name == 'pointnet2_seg_msg_radar':
        iou, acc = compute_metrics(np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0), dataset_test.seg_classes)
    print("Weighed Acc: {:.4f}".format(acc))
    print("Weighed Average IoU: {:.4f}".format(iou))
    print('Mean Loss: {:.4f}'.format(np.mean(losses)))
    print('='*40)
    print("Evaluating completed !")


if __name__ == '__main__':
    fire.Fire()