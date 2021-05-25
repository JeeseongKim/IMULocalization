import torch
from torch import nn
import random
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms
import torchvision
import cv2
cv2.ocl.setUseOpenCL(False)
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

class my_dataset(Dataset):
    def __init__(self, window_size):

        self.imu_x = []
        self.imu_y = []
        self.imu_z = []

        self.vicon_x = []
        self.vicon_y = []

        self.imu = []
        self.vicon= []

        self.dataset_filename = []
        self.dataset_filename_vicon = []

        self.window_size = window_size

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data/train/IMU/*.txt'))):
        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data_5_1/train/IMU/*.txt'))):
        for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data_1_5/train/IMU/*.txt'))):
            self.dataset_imu = []
            with open(filename) as f:
                self.dataset_imu.append(f.readlines())

            #if(len(self.dataset_imu)<5):
            self.dataset_imu = (self.dataset_imu[0])

            for i in range(len(self.dataset_imu)):
                self.imu_x.append(self.dataset_imu[i].split('\t')[3])
                self.imu_y.append(self.dataset_imu[i].split('\t')[4])
                self.imu_z.append(self.dataset_imu[i].split('\t')[5])
                self.dataset_filename.append(filename.split('/')[7].split('.')[0])

        np_imu_x = np.array(self.imu_x, dtype=np.float32)
        np_imu_y = np.array(self.imu_y, dtype=np.float32)
        np_imu_z = np.array(self.imu_z, dtype=np.float32)

        tensor_imu_x = torch.from_numpy(np_imu_x)
        tensor_imu_y = torch.from_numpy(np_imu_y)
        tensor_imu_z = torch.from_numpy(np_imu_z)

        tensor_imu_x = tensor_imu_x.view(tensor_imu_x.shape[0], 1)
        tensor_imu_y = tensor_imu_y.view(tensor_imu_y.shape[0], 1)
        tensor_imu_z = tensor_imu_z.view(tensor_imu_z.shape[0], 1)

        self.imu = torch.cat([tensor_imu_x, tensor_imu_y, tensor_imu_z], dim=1)

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data/train/Vicon/*.txt'))):
        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data_5_1/train/Vicon/*.txt'))):
        for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data_1_5/train/Vicon/*.txt'))):
            self.dataset_vicon = []
            with open(filename) as f:
                self.dataset_vicon.append(f.readlines())

            self.dataset_vicon = (self.dataset_vicon[0])

            for i in range(len(self.dataset_vicon)):
                self.vicon_x.append(self.dataset_vicon[i].split('\t')[5])
                self.vicon_y.append(self.dataset_vicon[i].split('\t')[6])
                self.dataset_filename_vicon.append(filename.split('/')[7].split('.')[0])

        np_vicon_x = np.array(self.vicon_x, dtype=np.float32)
        np_vicon_y = np.array(self.vicon_y, dtype=np.float32)

        tensor_vicon_x = torch.from_numpy(np_vicon_x)
        tensor_vicon_y = torch.from_numpy(np_vicon_y)

        tensor_vicon_x = tensor_vicon_x.view(tensor_vicon_x.shape[0], 1)
        tensor_vicon_y = tensor_vicon_y.view(tensor_vicon_y.shape[0], 1)

        self.vicon = torch.cat([tensor_vicon_x, tensor_vicon_y], dim=1)

        self.processing_imu = []
        for idx in range(self.imu.shape[0]):
            if(len(self.imu)-idx < window_size):
                #self.processing_imu.append(self.imu[idx:len(self.imu), :])
                break
            self.processing_imu.append(self.imu[idx:idx+window_size, :])

        self.processing_vicon = []
        for idx_vicon in range(self.vicon.shape[0]):
            if(len(self.vicon)-idx_vicon < window_size):
                #self.processing_vicon.append(self.vicon[idx:len(self.vicon), :])
                break
            self.processing_vicon.append(self.vicon[idx_vicon:idx_vicon+window_size, :])

        self.len = len(self.processing_imu)

    def __getitem__(self, index):
        return self.processing_imu[index], self.processing_vicon[index]

    def __len__(self):
        return len(self.processing_imu)


class my_test_dataset(Dataset):
    def __init__(self, imu_file, vicon_file, window_size):

        self.imu_x = []
        self.imu_y = []
        self.imu_z = []

        self.vicon_x = []
        self.vicon_y = []

        self.imu = []
        self.vicon= []

        self.dataset_filename = []
        self.dataset_filename_vicon = []

        self.window_size = window_size

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data/test/IMU/*.txt'))):
        #for filename in imu_file:
        filename=imu_file
        self.dataset_imu = []
        with open(filename) as f:
            self.dataset_imu.append(f.readlines())

        self.dataset_imu = (self.dataset_imu[0])

        for i in range(len(self.dataset_imu)):
            self.imu_x.append(self.dataset_imu[i].split('\t')[3])
            self.imu_y.append(self.dataset_imu[i].split('\t')[4])
            self.imu_z.append(self.dataset_imu[i].split('\t')[5])
            self.dataset_filename.append(filename.split('/')[7].split('.')[0])

        np_imu_x = np.array(self.imu_x, dtype=np.float32)
        np_imu_y = np.array(self.imu_y, dtype=np.float32)
        np_imu_z = np.array(self.imu_z, dtype=np.float32)

        tensor_imu_x = torch.from_numpy(np_imu_x)
        tensor_imu_y = torch.from_numpy(np_imu_y)
        tensor_imu_z = torch.from_numpy(np_imu_z)

        tensor_imu_x = tensor_imu_x.view(tensor_imu_x.shape[0], 1)
        tensor_imu_y = tensor_imu_y.view(tensor_imu_y.shape[0], 1)
        tensor_imu_z = tensor_imu_z.view(tensor_imu_z.shape[0], 1)

        self.imu = torch.cat([tensor_imu_x, tensor_imu_y, tensor_imu_z], dim=1)

        #for filename in (sorted(glob.glob('/home/jsk/IMUlocalization/data/test/Vicon/*.txt'))):
        #for filename in vicon_file:
        filename=vicon_file
        self.dataset_vicon = []
        with open(filename) as f:
            self.dataset_vicon.append(f.readlines())

        self.dataset_vicon = (self.dataset_vicon[0])

        for i in range(len(self.dataset_vicon)):
            self.vicon_x.append(self.dataset_vicon[i].split('\t')[5])
            self.vicon_y.append(self.dataset_vicon[i].split('\t')[6])
            self.dataset_filename_vicon.append(filename.split('/')[7].split('.')[0])

        np_vicon_x = np.array(self.vicon_x, dtype=np.float32)
        np_vicon_y = np.array(self.vicon_y, dtype=np.float32)

        tensor_vicon_x = torch.from_numpy(np_vicon_x)
        tensor_vicon_y = torch.from_numpy(np_vicon_y)

        tensor_vicon_x = tensor_vicon_x.view(tensor_vicon_x.shape[0], 1)
        tensor_vicon_y = tensor_vicon_y.view(tensor_vicon_y.shape[0], 1)

        self.vicon = torch.cat([tensor_vicon_x, tensor_vicon_y], dim=1)

        self.processing_imu = []
        for idx in range(0, len(self.imu)-200, window_size):
            self.processing_imu.append(self.imu[idx:idx+window_size, :])

        self.processing_vicon = []
        for idx_vicon in range(0, len(self.vicon) - 200, window_size):
            self.processing_vicon.append(self.vicon[idx_vicon:idx_vicon+window_size, :])

        self.len = len(self.processing_imu)

    def __getitem__(self, index):
        return self.processing_imu[index], self.processing_vicon[index]

    def __len__(self):
        return len(self.processing_imu)