import h5py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from vig_2classes import vig_ti_224_gelu
import numpy as np

def semantic_energy_loss(output_feature, nodes_label, regression, class_heat_maps,
                         lam1=2.0, lam2=0.02):
    
    #'output_feature': (batch_size,dim,H,W), a feature matrix used for calculating similarity, extracted from our network. 
    #                   Each pixel in the matrix represents a fixed-size region of the original image.(H and W are the height and width of the downsampled image)
    #'nodes_label':(batch_size,H,W), indicates which regions contain tumor cells. We set the value to 9 when a tumor cell is labeled.
    #'regression': (batch_size,num_class,height,width) prediction heatmap, num_class=2 in our set(labeled tumor or unlabeled)
    #'class_heat_maps': (batch_size,num_class,height,width) sparse ground truth
    # note height!=H width!=W


    B, dim, H, W = output_feature.shape
    nodes_label = nodes_label.reshape(B, -1)

    num_node = H * W
    output_feature = output_feature.reshape(B, dim, num_node)
    output_feature = output_feature.permute(0, 2, 1)


    total_regression_energy = torch.zeros(B, 2, 32, 32).cuda()
    mask = (nodes_label==9).cuda()
    mask = mask.reshape(B, 1, num_node)
    # calculating the number of labeled tumor cell
    num_instances = mask.float().sum(dim=2).reshape(B, 1)
    num_instances[num_instances == 0] = 1e-10

    x1_normalized = output_feature / torch.norm(output_feature, dim=2, keepdim=True)
    x2_normalized = output_feature / torch.norm(output_feature, dim=2, keepdim=True)

    cos_distance = torch.einsum('ijk,ilk->ijl', x1_normalized, x2_normalized)
    cos_distance=(1+cos_distance)/2


    regression_energy=mask*cos_distance
    regression_energy=torch.sum(regression_energy,dim=2)/num_instances
    regression_energy=1-regression_energy


    
    regression_energy[nodes_label == 9] = 1

    stretched_feature_loss=0
    for batch in range(B):
        labeled_index= mask[batch,:,:]==True   #labeled nodes
        unlabeled_index= mask[batch,:,:]==False #unlabeled nodes

        labeled_feature=output_feature[batch,labeled_index[0,:],:]
        unlabeled_feature=output_feature[batch,unlabeled_index[0,:],:]

        feature_distance=torch.cdist(labeled_feature, unlabeled_feature, p=2)
        feature_distance=torch.exp(-(feature_distance-10))        
        feature_distance=torch.mean(feature_distance)

        lam_labeled=num_instances[batch,:]/(num_node)
        

        stretched_feature_loss=stretched_feature_loss+lam_labeled*feature_distance
        


    stretched_feature_loss=stretched_feature_loss/B
    stretched_feature_loss=stretched_feature_loss.squeeze()

    

    total_regression_energy[:, 1] =regression_energy.reshape(B, H, W)
    total_regression_energy[:, 0] =regression_energy.reshape(B, H, W)

    #To resize total_regression_energy back to the original image size 
    #we have experimentally verified that there is essentially no difference between the
    #results using 'bilinear' and 'nearest' interpolation methods.
    total_regression_energy = F.interpolate(total_regression_energy, scale_factor=32, mode='bilinear')

    class_heat_maps = class_heat_maps.type(torch.float32)


    regression_loss=-class_heat_maps*torch.log(torch.clip(regression,1e-10,1.0))
    regression_loss=regression_loss*total_regression_energy
    regression_loss = torch.sum(regression_loss, dim=1)
    regression_loss = torch.mean(regression_loss)


    total_loss = lam1 * regression_loss + lam2 * stretched_feature_loss 

    return total_loss, regression_loss, stretched_feature_loss