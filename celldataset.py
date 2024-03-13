from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import glob
import h5py

from  skimage.feature import peak_local_max


class Cell_Dataset(Dataset):
    def __init__(self, data_root, gt_root, transform=None):
        self.root1 = data_root
        self.root2 = gt_root
        self.transform = transform
        self.image_files = sorted(os.listdir(data_root))
        self.h5_files = sorted(os.listdir(gt_root))

    def __getitem__(self, index):
        # load image
        img_path = os.path.join(self.root1, self.image_files[index])
        img = Image.open(img_path)

        # load ground truth data
        h5_path = os.path.join(self.root2, self.h5_files[index])

        with h5py.File(h5_path, 'r') as hf:
           
            class_heat_map = np.array(hf.get('sparse_class_heat_map'))
            full_calss_heat_map=np.array(hf.get('class_heat_map'))
            nodes_label_formal=np.array(hf.get('sparse_target'))

        label_coordinates = peak_local_max(class_heat_map[1], min_distance=12, exclude_border=6 // 2)

        #Here are the code about how to generate node label, 32 is the downsample rate.
        #nodes_label=np.zeros(nodes_label_formal.shape)
        #for i in range(label_coordinates.shape[0]):
        #    x=int(label_coordinates[i][0]/32)
        #    y=int(label_coordinates[i][1]/32)
        #    nodes_label[x][y]=9

        if self.transform is not None:
            img = self.transform(img)

        return img, class_heat_map, full_calss_heat_map, nodes_label

    def __len__(self):
        return len(self.image_files)
