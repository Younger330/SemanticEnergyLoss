import argparse

import torch
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.optim.lr_scheduler as lr_scheduler

from semantic_energy_loss import semantic_energy_loss

from vig_2classes import vig_ti_224_gelu,vig_s_224_gelu

from gcn_lib.torch_edge import DenseDilatedKnnGraph

from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from celldataset import Cell_Dataset
import torchvision.transforms as transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def acc(full_target, predict):
    matchs = torch.sum(torch.eq(full_target, predict))
    return matchs / full_target.shape[1]

def get_arguments():
    parser = argparse.ArgumentParser(description="semantic_energy")

    parser.add_argument("--img_dir", type=str, default='/home/data/xuzhengyang/Her2/img_2classes/train', help="your training image path")
    parser.add_argument("--gt_dir", type=str, default='/home/data/xuzhengyang/Her2/ground_truth_2classes_new/train')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--lam1", type=float, default=2.0)
    parser.add_argument("--lam2", type=float, default=0.05)

    return parser


def main():

    parser = get_arguments()
    print(parser)

    args = parser.parse_args()
    

    train_dataset = Cell_Dataset(data_root=args.img_dir, gt_root=args.gt_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vig_model=vig_s_224_gelu()

    # pre_trained
    # model_state_dict=torch.load('/home/xuzhengyang/vig_model_2classes_epoch_2023-06-16_11_52_05.pkl')
    # vig_model.load_state_dict(model_state_dict)


    vig_model.cuda()

    epochs = args.epoch
    vig_model.train()
    optimizer = torch.optim.Adam(vig_model.parameters(), lr=0.01)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)


    for epoch in tqdm(range(epochs),desc='Epoch'):
        epoch_loss=0
        mean_regression=0
        mean_stretched_feature=0
        for images, class_heat_maps, position_heat_maps, nodes_label in tqdm(train_dataloader):
            optimizer.zero_grad()
            images=images.cuda()
            class_heat_maps=torch.clip(class_heat_maps,0.0,1.0)
            class_heat_maps=class_heat_maps.cuda()
            
            output_feature,regression = vig_model.forward(images)
            
            loss, regression_loss, stretched_feature_loss = semantic_energy_loss(output_feature, nodes_label, regression, class_heat_maps,lam1=args.lam1,lam2=args.lam2)
            
            loss.backward()

            optimizer.step()
            scheduler.step()


            epoch_loss+=loss
            mean_regression=mean_regression+regression_loss
            mean_stretched_feature=mean_stretched_feature+stretched_feature_loss

        epoch_loss=epoch_loss/len(train_dataloader)
        mean_regression=mean_regression/len(train_dataloader)
        mean_stretched_feature=mean_stretched_feature/len(train_dataloader)

        tqdm.write(f"Epoch {epoch}: Loss={epoch_loss:.4f},regression_Loss={mean_regression:.4f},stretched_feature_loss={mean_stretched_feature:.4f}")

        if epoch%50==0:
            now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
            save_name=f'vig_model_2classes_epoch_{now}.pkl'
            torch.save(vig_model.state_dict(), save_name)


    now=datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    save_name=f'vig_model_2classes_{now}.pkl'
    torch.save(vig_model.state_dict(), save_name)


if __name__ == '__main__':
    main()



