import os
import numpy as np
import sys
import time
import torch
import random
import pandas as pd
from torch import nn
from tqdm import tqdm
import torchio as tio
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import MedData_Load
# Self defined functions
from models.OCLNet import OCL
from utils.func_322 import metric_test, Dice_Loss
from torch.utils.tensorboard import SummaryWriter

GLOBAL_SEED = 125
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.set_num_threads(4)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# Change Parameters Here
def hyperparameter_setting(model_type, learning_rate, weight_decay, resize_size, batch_size, time_set):
    class hp:
        def __init__(self, model_type, learning_rate, weight_decay, resize_size, batch_size, time_set):
            # Parameters
            self.model_type = model_type
            self.total_epochs = 200
            self.epochs_per_checkpoint = 40
            self.batch_size = batch_size
            self.init_lr = learning_rate
            self.weight_decay = weight_decay
            self.resize_size = resize_size
            self.in_class = 1
            self.out_class = 1
            self.model_id = self.model_type + '_' + str(self.batch_size) + '_' + str(self.init_lr) + '_' + str(self.weight_decay) + '_' + str(self.resize_size) + '_' + str(self.total_epochs) + '_' + time_set
            
            # Output Folders
            self.ckpt_dir = './OUTPUTS/logs13/'+ self.model_type + '/' + self.model_id
            self.output_dir = './OUTPUTS/results13/' + self.model_type + '/' + self.model_id
            
            # Data Folders
            self.image_dir = './datasets/imagenii160crop'
            self.label_dir = './datasets/labelnii160crop'
            
            self.csv_dir = './datasets/160_train.csv'
            
            self.fold_arch = '*.nii'
            self.save_arch = '.nii'
    
    return hp(model_type, learning_rate, weight_decay, resize_size, batch_size, time_set)

# Keep tracking the selected GPU's memory
def run_experiment(hp, flag, gpu_alert, gpu_idx):
    run_flag = False
    while(not run_flag):
        nvidia_smi = os.popen('nvidia-smi | grep %').readlines()
        for i in range(2):
            memory_used = int(nvidia_smi[i].split('|')[2].split('M')[0].strip())
            if memory_used < gpu_alert and (i == gpu_idx):
                run_flag = True
                # os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                print(flag + "ing Model: " + hp.model_id)

                if flag == 'Train':
                    train(hp)
                elif flag == 'Test':
                    test(hp)
                else:
                    print('Wrong flag value, either <Train> or <Test>')
                break
    return
writer = SummaryWriter('./tensorboard')
# Data Loading Class


def train(hp):
    
    os.makedirs(hp.ckpt_dir, exist_ok = True)
    id_list = []
    lc_list = []
    tc_list = []
    shape_list = []
    zero_id_list = []
    prec_list = []
    dice_list = []
    jacc_list = []
    aucs_list = []

    if model_type == 'OCL':
        model = OCL(in_channels = hp.in_class, out_channels = hp.out_class)
    else:
        print("No Such Model Type!")

    optimizer = torch.optim.Adam(model.parameters(), lr = hp.init_lr, weight_decay = hp.weight_decay)
    model.cuda()

    print("Loading Image ...")
    print(time.asctime(time.localtime(time.time())))
    train_dataset = MedData_Load('Train', hp.image_dir, hp.label_dir, hp.folds_csv_dir, hp.fold_arch, hp.resize_size, hp.fold)
    train_loader = DataLoader(train_dataset.dataset, batch_size = hp.batch_size, shuffle = True, num_workers = 4)
    BCELoss = nn.BCELoss()
    DCLoss = Dice_Loss()
    
    total_epochs = hp.total_epochs

    print("Start Training ...")
    print(time.asctime(time.localtime(time.time())))
    for epoch in tqdm(range(1, total_epochs + 1), ncols=80):
        epoch_loss = 0.0
        model.train()
        for batch_train in train_loader:
            img = batch_train['image']['data'].type(torch.FloatTensor).cuda().requires_grad_(True)
            lab = batch_train['label']['data'].type(torch.FloatTensor).cuda().requires_grad_(True)

            optimizer.zero_grad()
            img_shape = tuple(tensor.item() for tensor in batch_train['image_shape'])
            out = model(img)
            out_sig = torch.sigmoid(out)
            
            tloss = BCELoss(out_sig, lab) + DCLoss(out_sig, lab)
            
            tloss.backward()
            optimizer.step()
            epoch_loss += tloss.item()
            if epoch%10 == 0:
                with torch.no_grad():                
                    prd = out_sig.clone()
                    prd = (prd >= 0.5).int()
                    prd_shape = tuple(torch.squeeze(prd).shape)
                    prec, dice, jacc, aucs = metric_test(lab.cpu(), prd.cpu())

                    coordinates = torch.nonzero(torch.squeeze(prd), as_tuple=False)
                    if coordinates.numel() > 0:
                        low_coords = tuple(torch.min(coordinates, dim=0).values.tolist())
                        top_coords = tuple(torch.max(coordinates, dim=0).values.tolist())
                        
                        scale_factors = [orig_size / small_size for orig_size, small_size in zip(img_shape, prd_shape)]

                        original_low = tuple(int(coord * scale_factor) for coord, scale_factor in zip(low_coords, scale_factors))
                        original_top = tuple(int(coord * scale_factor) for coord, scale_factor in zip(top_coords, scale_factors))
                    else:
                        original_low = (0, 0, 0)
                        original_top = (0, 0, 0)
                    
                    lc_list.append(str(original_low))
                    tc_list.append(str(original_top))
                    shape_list.append(str(img_shape))
                    prec_list.append(prec)
                    dice_list.append(dice)
                    print('dice:{}'.format(dice))
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        if epoch % hp.epochs_per_checkpoint == 0:
            print("its saving for epoch ",epoch)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(hp.ckpt_dir, f'Checkpoint_epoch_{epoch}.pt'),
            )

def test(hp):
    os.makedirs(hp.output_dir, exist_ok = True)
    
    if model_type == 'OCL':
        model = OCL(in_channels = hp.in_class, out_channels = hp.out_class)
    else:
        print("No Such Model Type!")
    
    ckpt = torch.load(os.path.join(hp.ckpt_dir, 'Checkpoint.pt'), map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt["model"])
    model.cuda()

    test_dataset = MedData_Load('Test', hp.image_dir, hp.label_dir, hp.folds_csv_dir, hp.fold_arch, hp.resize_size, hp.fold)
    test_loader = DataLoader(test_dataset.dataset, batch_size = 1)
    
    model.eval()

    id_list = []
    lc_list = []
    tc_list = []
    shape_list = []
    zero_id_list = []
    prec_list = []
    dice_list = []
    jacc_list = []
    aucs_list = []

    with torch.no_grad():
        for batch in test_loader:
            img = batch['image']['data'].type(torch.FloatTensor).cuda()
            # lab = batch['label']['data'].type(torch.FloatTensor).cuda()
            id_list.append(batch['ID'][0])
            img_shape = tuple(tensor.item() for tensor in batch['image_shape'])

            out = model(img)
            sig = torch.sigmoid(out)

            prd = sig.clone()
            prd = (prd >= 0.5).int()
            prd_shape = tuple(torch.squeeze(prd).shape)

            # prec, dice, jacc, aucs = metric_test(lab.cpu(), prd.cpu())

            # coordinates = torch.nonzero(torch.squeeze(prd), as_tuple=False)
            # if coordinates.numel() > 0:
            #     low_coords = tuple(torch.min(coordinates, dim=0).values.tolist())
            #     top_coords = tuple(torch.max(coordinates, dim=0).values.tolist())
                
            #     scale_factors = [orig_size / small_size for orig_size, small_size in zip(img_shape, prd_shape)]

            #     original_low = tuple(int(coord * scale_factor) for coord, scale_factor in zip(low_coords, scale_factors))
            #     original_top = tuple(int(coord * scale_factor) for coord, scale_factor in zip(top_coords, scale_factors))
            # else:
            #     original_low = (0, 0, 0)
            #     original_top = (0, 0, 0)
            #     zero_id_list.append(batch['ID'][0])
            
            # lc_list.append(str(original_low))
            # tc_list.append(str(original_top))
            # shape_list.append(str(img_shape))
            # prec_list.append(prec)
            # dice_list.append(dice)
            # jacc_list.append(jacc)
            # aucs_list.append(aucs)
            affine = batch['image']['affine'][0]
            source_image = tio.ScalarImage(tensor=img[0].cpu(),affine = affine)
            source_image.save(os.path.join(hp.output_dir, batch['ID'][0]+'-source'+hp.save_arch))

            predict_image = tio.ScalarImage(tensor=prd[0].cpu(),affine = affine)
            predict_image.save(os.path.join(hp.output_dir, batch['ID'][0]+'-predict'+hp.save_arch))

            # source_image = tio.ScalarImage(tensor=lab[0].cpu(),affine = affine)
            # source_image.save(os.path.join(hp.output_dir, batch['ID'][0]+'-label'+hp.save_arch))
    
    # save_data = {'ID': id_list, 'Image_Shape': shape_list, 
    #              'Low_Coords': lc_list, 'Top_Coords': tc_list,
    #              'Prec': prec_list, 'Dice': dice_list,
    #              'IoU': jacc_list, 'AUC': aucs_list}
    # save_df = pd.DataFrame(save_data)
    
    # print(f"Model Prec: {sum(prec_list)/len(test_loader):.4f}, Dice: {sum(dice_list)/len(test_loader):.4f}, IoU: {sum(jacc_list)/len(test_loader):.4f} and AUC: {sum(aucs_list)/len(test_loader):.4f}.")

    # excel_file_path = os.path.join(hp.output_dir, 'Coords_and_Results.xlsx')
    # save_df.to_excel(excel_file_path, index=False)

    return

if __name__ == '__main__':
    if len(sys.argv) != 8:
        print("Usage: python <Script Name> <GPU ID> <Model Type> <Learning Rate> <Weight Decay> <Time> <Train/Test Flag>")
        sys.exit(1)

    gpu_alert = 32000
    resize_size = 96
    gpu_idx = int(sys.argv[1])
    model_type = sys.argv[2]
    learning_rate = float(sys.argv[3])
    weight_decay = float(sys.argv[4])
    time_set = sys.argv[5]
    flag = sys.argv[6]
    batch_size = 1

    hp = hyperparameter_setting(model_type, learning_rate, weight_decay, resize_size, batch_size, time_set)
    run_experiment(hp, flag, gpu_alert, gpu_idx)