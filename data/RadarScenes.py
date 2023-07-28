import h5py
import json
import numpy as np
import os
import sys
import math
import warnings
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.provider import augment_pc, pc_normalize, random_point_dropout, convert_to1


class RadarScenes():
    def __init__(self, data_root, split, npoints, combined_frame_num, augment=True, dp=False, normalize=False):
        assert(split == 'train' or split == 'validation')
        self.npoints = npoints
        self.combined_frame_num = combined_frame_num
        self.augment = augment
        self.dp = dp
        self.normalize = normalize

        with open(os.path.join(data_root, 'sequences.json'), 'r') as f:
            sequences = json.load(f)
            # generate train and validation lists
            train_radar_data_lists = []
            validation_radar_data_lists = []
            train_scenes_lists = []
            validation_scenes_lists = []
            for index in range(len(sequences['sequences'])):
                if sequences['sequences']['sequence_'+str(index+1)]['category'] == 'train':
                    # append path to train list
                    train_radar_data_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'radar_data.h5'))
                    train_scenes_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'scenes.json'))
                elif sequences['sequences']['sequence_'+str(index+1)]['category'] == 'validation':
                    # append path to validation list
                    validation_radar_data_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'radar_data.h5'))
                    validation_scenes_lists.append(os.path.join(data_root, 'sequence_'+str(index+1), 'scenes.json'))

        self.h5_file_lists = []
        self.json_file_lists = []
        if split == 'train':
            self.h5_file_lists.extend(train_radar_data_lists)
            self.json_file_lists.extend(train_scenes_lists)
            # reduce the number of files for debug
            # self.h5_file_lists = self.h5_file_lists[:1]
            # self.json_file_lists = self.json_file_lists[:1]
        elif split == 'validation':
            self.h5_file_lists.extend(validation_radar_data_lists)
            self.json_file_lists.extend(validation_scenes_lists)
            # reduce the number of files for debug
            # self.h5_file_lists = self.h5_file_lists[:1]
            # self.json_file_lists = self.json_file_lists[:1]

        self.file_num = len(self.h5_file_lists)
        # number of combined frames in all files
        self.file_combined_frame_num_list = self.get_file_combined_frame_num_list()
        self.seg_classes = {'Scenes': [0,1,2,3,4,5]}
        self.caches = {}

    def get_file_num(self):
        return self.file_num

    def get_frame_num(self, file_index):
        # this function is used to get the number of frames in a file
        json_file = json.load(open(self.json_file_lists[file_index], 'r'))
        return len(json_file['scenes'])
    
    def get_combined_frame_num(self, file_index):
        # this function is used to get the number of combined frames in a file
        json_file = json.load(open(self.json_file_lists[file_index], 'r'))
        return math.ceil(len(json_file['scenes'])/self.combined_frame_num)

    def label_translation(self, labels):
        # original labels are:
        # 0: CAR
        # 1: LARGE_VEHICLE
        # 2: TRUCK
        # 3: BUS
        # 4: TRAIN
        # 5: BICYCLE
        # 6: MOTORIZED_TWO_WHEELER
        # 7: PEDESTRIAN
        # 8: PEDESTRIAN_GROUP
        # 9: ANIMAL
        # 10: OTHER
        # 11: STATIC
        # target labels are:
        # 0: CAR
        # 1: PEDESTRIAN
        # 2: PEDESTRIAN_GROUP
        # 3: TWO_WHEELER
        # 4: LARGE_VEHICLE
        # 5: STATIC
        for i in range(len(labels)):
            if labels[i] == 0:
                labels[i] = 0
            elif labels[i] == 1:
                labels[i] = 4
            elif labels[i] == 2:
                labels[i] = 4
            elif labels[i] == 3:
                labels[i] = 4
            elif labels[i] == 4:
                labels[i] = 4
            elif labels[i] == 5:
                labels[i] = 3
            elif labels[i] == 6:
                labels[i] = 3
            elif labels[i] == 7:
                labels[i] = 1
            elif labels[i] == 8:
                labels[i] = 2
            elif labels[i] == 9:
                labels[i] = 1
            elif labels[i] == 10:
                labels[i] = 5
            elif labels[i] == 11:
                labels[i] = 5
        return labels
    
    def get_file_combined_frame_num_list(self):
        # this function will return number of all combined frames in all files
        num_processes = 16
        file_combined_frame_num_list = []
        with Pool(num_processes) as p:
            file_combined_frame_num_list = p.map(self.get_combined_frame_num, range(self.file_num))
        return file_combined_frame_num_list
        
    def get_index_position(self, index):
        # this function will return the file name, timestamp, frame index in the file, and total number of frames in the file
        file_count = 0
        frame_index = None

        for file_index, num in enumerate(self.file_combined_frame_num_list):
            if index <= file_count + (num - 1):
                frame_index = index - file_count
                break
            else:
                file_count += num

        if frame_index is not None:
            json_file = json.load(open(self.json_file_lists[file_index], 'r'))
            frame_list = list(json_file['scenes'].keys())

            chosen_frames = frame_list[frame_index*self.combined_frame_num:(frame_index+1)*self.combined_frame_num]

            return self.json_file_lists[file_index], chosen_frames[0], frame_index*self.combined_frame_num, len(frame_list)

    def __len__(self):
        return sum(self.file_combined_frame_num_list)
        
    def __getitem__(self, index):
        # index will be converted to file_index and frame_index according to file_combined_frame_num_list
        # initial file_count and frame_index
        file_count = 0
        frame_index = None

        for file_index, num in enumerate(self.file_combined_frame_num_list):
            if index <= file_count + (num - 1):
                frame_index = index - file_count
                break
            else:
                file_count += num

        if frame_index is not None:
            if index in self.caches:
                xyz_points, labels = self.caches[index]
            else:
                h5_file = h5py.File(self.h5_file_lists[file_index], 'r')
                radar_data = h5_file['radar_data'][:]

                json_file = json.load(open(self.json_file_lists[file_index], 'r'))
                frame_list = list(json_file['scenes'].keys())

                chosen_frames = frame_list[frame_index*self.combined_frame_num:(frame_index+1)*self.combined_frame_num]
                # if chosen_frames is not enough, it means the frame_index is the last frame in the file
                # just use the data which number of self.combined_frame_num from the end of the file
                if len(chosen_frames) < self.combined_frame_num:
                    chosen_frames = frame_list[-self.combined_frame_num:-1]
                    chosen_frames.append(frame_list[-1])
                # convert to int
                chosen_frames = [int(ts) for ts in chosen_frames]
                chosen_data = radar_data[np.isin(radar_data['timestamp'], chosen_frames)]
                # convert to xyz points, labels
                # xyz points structure is:
                # [x,y,z,compensated Doppler velocity,RCS,0]*n numpy.ndarray, dtype=float32
                # labels structure is:
                # [point_type, point_type, ...] numpy.ndarray, dtype=int32
                xyz_points = np.zeros((len(chosen_data), 5)).astype(np.float32)
                xyz_points[:, 0] = chosen_data[:]['x_cc']
                xyz_points[:, 1] = chosen_data[:]['y_cc']
                # keep z as 0
                xyz_points[:, 3] = chosen_data[:]['vr_compensated']
                xyz_points[:, 4] = chosen_data[:]['rcs']
                
                labels = chosen_data[:]['label_id'].astype(np.int32)
                labels = self.label_translation(labels)

                if self.normalize:
                    xyz_points[:, :3] = pc_normalize(xyz_points[:, :3])
                    # 4,5 convert_to1
                    xyz_points[:, 3] = convert_to1(xyz_points[:, 3])
                    xyz_points[:, 4] = convert_to1(xyz_points[:, 4])
                if self.augment:
                    xyz_points = augment_pc(xyz_points)
                if self.dp:
                    xyz_points = random_point_dropout(xyz_points)
                self.caches[index] = xyz_points, labels

            choice = np.random.choice(len(xyz_points), self.npoints, replace=True)
            xyz_points = xyz_points[choice, :]
            # replace the nan in xyz_points with 0
            xyz_points = np.nan_to_num(xyz_points)
            
            labels = labels[choice]
            return xyz_points, labels
        else:
            warnings.warn("Index out of range.")
            return None
        
    def plot_combined_frame(self, index):
        import matplotlib.pyplot as plt
        xyz_points, labels = self.__getitem__(index)

        plt.scatter(xyz_points[:, 1], xyz_points[:, 0], c=labels, s=5)
        # reverse the x axis
        plt.xlim(plt.xlim()[::-1])
        plt.show()


if __name__ == '__main__':
    # combined_frame_num is the number of frames combined into one frame for more points
    radarscenes = RadarScenes(data_root='./data/RadarScenes', split='train', npoints=3072, combined_frame_num=28, augment=True)
    print(radarscenes.__len__())
    print(radarscenes.__getitem__(0))
    print(radarscenes.get_index_position(4))
    # plot the xyz_points according to the labels and x_cc, y_cc
    #radarscenes.plot_combined_frame(0)