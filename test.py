import torch
import torch.nn.functional as F

# from energy_loss1 import semantic_energy_loss
# from vig2 import vig_ti_224_gelu,vig_s_224_gelu
from vig_2classes import vig_ti_224_gelu,vig_s_224_gelu
# from vig_2classes3 import vig_ti_224_gelu,vig_s_224_gelu

from torch.utils.data import DataLoader
from celldataset import Cell_Dataset

import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib


import h5py
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from  skimage.feature import peak_local_max
import numpy as np

import numpy as np
import scipy.spatial as S
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def binary_match(pred_points, gd_points, threshold_distance=15):
    dis = S.distance_matrix(pred_points, gd_points)
    connection = np.zeros_like(dis)
    connection[dis < threshold_distance] = 1
    graph = csr_matrix(connection)
    res = maximum_bipartite_matching(graph, perm_type='column')
    right_points_index = np.where(res > 0)[0]
    right_num = right_points_index.shape[0]

    matched_gt_points = res[right_points_index]
    text=np.unique(matched_gt_points)



    if (np.unique(matched_gt_points)).shape[0] != (matched_gt_points).shape[0]:
        import pdb;
        pdb.set_trace()

    return right_num, right_points_index


threshold=0.7
img_root='/home/data/xuzhengyang/Her2/img_2classes/val'
model_state_dict=torch.load('')
# print(model_state_dict)

weights_dict={}
for k, v in model_state_dict.items():
    new_k = k.replace('module.', '') if 'module' in k else k
    weights_dict[new_k] = v


vig_model=vig_s_224_gelu()

vig_model.load_state_dict(weights_dict)
vig_model.cuda()
vig_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

len=0
total_precision=0
total_recall=0
for folder1 in os.listdir(img_root):
    len=len+1
    img_path = os.path.join(img_root, folder1)
    h5_path=img_path.replace('img_2classes', 'ground_truth_2classes_new').replace('.png', '.h5')
    
    images = Image.open(img_path)
    images = transform(images)
    images=images.cuda()
    C,H,W=images.shape
    images=images.reshape(1,C,H,W)

    with h5py.File(h5_path, 'r') as hf:
        class_heat_map = np.array(hf.get('class_heat_map'))
        nodes_label=np.array(hf.get('sparse_target'))
        sparse_class_heat_map=np.array(hf.get('sparse_class_heat_map'))
    label_coordinates=peak_local_max(class_heat_map[1], min_distance=12,  exclude_border=6 // 2)
    sparse_label_coordinates=peak_local_max(sparse_class_heat_map[1], min_distance=6,  exclude_border=6 // 2)


    with torch.no_grad():
        output_feature,regression = vig_model.forward(images)
    regression=regression.reshape(2,H,W)

    regression=regression.cpu().detach().numpy()

    cell_mask=regression[1]
    cell_mask[cell_mask<threshold]=0
    min_len=6
    coordinates=peak_local_max(cell_mask, min_distance=min_len,  exclude_border=min_len // 2)

    right_num,_=binary_match(coordinates,label_coordinates)

    precision=right_num/(coordinates.shape[0])
    recall=right_num/(label_coordinates.shape[0])
    print("precision is "+str(precision))
    print("recall is "+str(recall))
    total_precision=total_precision+precision
    total_recall=total_recall+recall


total_precision=total_precision/len
total_recall=total_recall/len
F1_score=(2*total_precision*total_recall)/(total_precision+total_recall)

print("====================")
print(len)
print("total precision is "+str(total_precision))
print("total recall is "+str(total_recall))
print("F1 score is "+str(F1_score))
