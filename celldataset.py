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
        # print(img_path)

        # load ground truth data
        h5_path = os.path.join(self.root2, self.h5_files[index])
        # print(h5_path)

        with h5py.File(h5_path, 'r') as hf:
            # class_heat_map = np.array(hf.get('class_heat_map'))
            # position_heat_map = np.array(hf.get('position_heat_map'))
            class_heat_map = np.array(hf.get('sparse_class_heat_map'))
            #position_heat_map = np.array(hf.get('sparse_position_heat_map'))
            full_calss_heat_map=np.array(hf.get('class_heat_map'))
            nodes_label_formal=np.array(hf.get('sparse_target'))

        label_coordinates = peak_local_max(class_heat_map[1], min_distance=12, exclude_border=6 // 2)
        # sparse_label_coordinates=peak_local_max(class_heat_map[1], min_distance=6,  exclude_border=6 // 2)
        # print(sparse_label_coordinates)
        nodes_label=np.zeros(nodes_label_formal.shape)

        # for i in range(sparse_label_coordinates.shape[0]):
        #     x=int(sparse_label_coordinates[i][0]/32)
        #     y=int(sparse_label_coordinates[i][1]/32)
        #     nodes_label[x][y]=9

        for i in range(label_coordinates.shape[0]):
            x=int(label_coordinates[i][0]/32)
            y=int(label_coordinates[i][1]/32)
            nodes_label[x][y]=9

        if self.transform is not None:
            img = self.transform(img)

        return img, class_heat_map, full_calss_heat_map, nodes_label

    def __len__(self):
        return len(self.image_files)


# transform = transforms.Compose([
#     # transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# dataset = Cell_Dataset(data_root='/home/xuzhengyang/code/vig_pytorch/data/img', gt_root='/home/xuzhengyang/code/vig_pytorch/data/ground_truth', transform=transform)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# for images, class_heat_maps, position_heat_maps in dataloader:
#     print(images.shape)
#     print(class_heat_maps.shape)
#     print(position_heat_maps.shape)
'''
    torch.Size([10, 10, 1024, 1024])   images
    torch.Size([10, 2, 1024, 1024])    class_heat_maps
    torch.Size([10, 10, 1024, 1024])   position_heat_maps
    '''