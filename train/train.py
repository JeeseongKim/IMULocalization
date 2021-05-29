import numpy as np
from tqdm import tqdm
from utils import *
from loss import *
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore")
# from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader
import time

from model.IMUTransformer import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom

def train():
    window_size = 200  # 200
    hidden_dim = 256

    num_epochs = 500
    batch_size = 128
    lr_drop = 300
    learning_rate = 1e-4
    weight_decay = 1e-5

    dtype = torch.FloatTensor

    vis = visdom.Visdom()
    plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))

    start = time.time()
    print("Training")
    #model definition
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    #IMUTransformer = IMU_Transformer(ntoken=window_size, ninp=256, nhid=256, nhead=8, nlayers=6, dropout=0.5).cuda()
    #IMUTransformer = IMU_Transformer_dist(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.5).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()
    optimizer_IMU = torch.optim.AdamW(IMUTransformer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_IMU, lr_drop)

    #load ckpt

    if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_model.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/epoch20.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_model.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/epoch20.pth")
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])
        optimizer_IMU.load_state_dict(checkpoint['optimizer_IMU'])
        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])


    #dataset = my_dataset(window_size=window_size)
    #dataset = my_dataset_input6(window_size=window_size)
    dataset = my_dataset_gyroz(window_size=window_size)
    #dataset = my_dataset_dist(window_size=window_size)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    #train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    get_loss = pos_loss()

    #SaveLossTxt = open("MyLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            cur_imu, cur_trg = data #(b,10,3)

            #src = cur_imu.cuda()*9.8 #m/s^2
            src = cur_imu.cuda() #m/s^2
            trg = cur_trg[:, 0:cur_trg.shape[1]-1, :].cuda() * 0.001 #mm -> m
            gt = cur_trg[:, 1:cur_trg.shape[1], :].cuda() * 0.001 #mm -> m
            #trg = cur_trg[:, 0:cur_trg.shape[1]-1, 2].cuda() * 0.001 #mm -> m
            #gt = cur_trg[:, 1:cur_trg.shape[1], 2].cuda() * 0.001 #mm -> m
            #xy_pos = IMUTransformer(src, trg)

            #if(i==0):
            #    transformer_target = trg
            #else:
            #    transformer_target = xy_pos

            #xy_pos = IMUTransformer(src, transformer_target)
            xy_pos = IMUTransformer(src, trg)

            loss = get_loss(xy_pos, gt)
            print("My loss", '%.8f' % loss.item())

            optimizer_IMU.zero_grad()
            loss.backward()
            #loss.backward()
            torch.autograd.set_detect_anomaly(True)
            optimizer_IMU.step()

            running_loss = running_loss + loss

        #np.savetxt(SaveLossTxt, running_loss)
        torch.save({
            'IMUTransformer': IMUTransformer.module.state_dict(),
            'optimizer_IMU': optimizer_IMU.state_dict(),
            'lr_scheduler_optimizer1': lr_scheduler_optimizer1.state_dict(),
        }, "/home/jsk/IMUlocalization/ckpt/train_model.pth")

        vis.line(Y=[running_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plot_all, update='append')
        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))

'''
def test():
    print("Testing")
    start = time.time()

    GTtxt = open("GT.txt", 'w')
    XYtxt = open("XY.txt", 'w')

    #model definition
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()

    #checkpoint save
    #if os.path.exists("/home/jsk/IMUlocalization/train_model.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_1_test_5.pth"):
    if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth"):
        print("-----Loading Checkpoint-----")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_1_test_5.pth")
        checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth")
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])

    #imu_file = "/home/jsk/IMUlocalization/data/test/IMU/imu_1cycle_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data/test/Vicon/vicon_1cycle_01.txt"

    imu_file = "/home/jsk/IMUlocalization/data_5_1/test/IMU/imu_1cycle_03.txt"
    vicon_file = "/home/jsk/IMUlocalization/data_5_1/test/Vicon/vicon_1cycle_03.txt"

    #imu_file = "/home/jsk/IMUlocalization/data_1_5/test/IMU/imu_5cycle_05.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data_1_5/test/Vicon/vicon_5cycle_05.txt"

    dataset = my_test_dataset(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    test_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    test_loss = 0
    final_gt = []
    final_xy = []
    #for i, data in enumerate(tqdm(train_loader)):
    for i, data in enumerate((test_loader)):
        cur_imu, cur_vicon = data #(b,10,3)

        src = cur_imu.cuda() * 9.8 #m/s^2
        gt = cur_vicon.cuda() * 0.001 #mm -> m

        xy_pos = IMUTransformer(src, gt)

        final_gt.append(gt)
        final_xy.append(xy_pos)

        get_loss = pos_loss()
        loss = get_loss(xy_pos, gt)

        print("My loss", '%.8f' % loss.item())
        test_loss = test_loss + loss

        #result = '{:.4f} \t{:.4f} \t{:.4f}\t{:.4f} \t{:.4f}\t{:.4f}\n'.format()

    for j in range(len(final_gt)):
        final_gt[j] = final_gt[j].detach().cpu().numpy()
        final_xy[j] = final_xy[j].detach().cpu().numpy()

        if (j != len(final_gt) - 1):
            np.savetxt(GTtxt, final_gt[j][0])
            np.savetxt(GTtxt, final_gt[j][1])
            np.savetxt(XYtxt, final_xy[j][0])
            np.savetxt(XYtxt, final_xy[j][1])

        if(j==len(final_gt)-1):
            if(len(final_gt[j])==1):
                np.savetxt(GTtxt, final_gt[j][0])
                np.savetxt(XYtxt, final_xy[j][0])
            else:
                np.savetxt(GTtxt, final_gt[j][0])
                np.savetxt(GTtxt, final_gt[j][1])
                np.savetxt(XYtxt, final_xy[j][0])
                np.savetxt(XYtxt, final_xy[j][1])
'''


def test():
    print("Testing")
    window_size = 200  # 200

    GTtxt = open("GT.txt", 'w')
    XYtxt = open("XY.txt", 'w')

    #model definition
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0).cuda()
    #IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=False).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()

    #checkpoint save
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/epoch20.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/epoch100_dropout0.5.pth"):
    if os.path.exists("/home/jsk/IMUlocalization/ckpt/epoch120_withgyro.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_model.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/test.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_1_test_5.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth"):
        print("-----Loading Checkpoint-----")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/epoch20.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/epoch100_dropout0.5.pth")
        checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/epoch120_withgyro.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_model.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/test.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_1_test_5.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth")
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])

    #imu_file = "/home/jsk/IMUlocalization/data/test/IMU/imu_1cycle_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data/test/Vicon/vicon_1cycle_01.txt"

    #imu_file = "/home/jsk/IMUlocalization/parsed_data/IMU/parsed_imu_1cycle_01_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/parsed_data/Vicon/parsed_vicon_1cycle_01_01.txt"

    imu_file = "/home/jsk/IMUlocalization/data_5_1/test/IMU/imu_1cycle_01.txt"
    vicon_file = "/home/jsk/IMUlocalization/data_5_1/test/Vicon/vicon_1cycle_01.txt"

    #imu_file = "/home/jsk/IMUlocalization/data_1_5/test/IMU/imu_5cycle_05.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data_1_5/test/Vicon/vicon_5cycle_05.txt"

    #dataset = my_test_dataset(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    #dataset = my_test_dataset_modified(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    dataset = my_test_dataset_gyroz(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    #dataset = my_test_dataset_modified_inp6(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    #dataset = my_dataset(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    test_loss = 0
    final_gt = []
    final_xy = []
    get_loss = pos_loss()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            cur_imu, cur_trg = data

            src = cur_imu.cuda()
            trg = cur_trg[:, 0:cur_trg.shape[1] - 1, :].cuda() * 0.001  # mm -> m (1~199)
            gt = cur_trg[:, 1:cur_trg.shape[1], :].cuda() * 0.001  # mm -> m (2~200)

            #gyro_z = src[:, :, 2]

            #xy_pos = IMUTransformer(src, trg)
            #if (i % window_size == 0) or (i % window_size == window_size-1) or (i == len(test_loader)-window_size):
            #if (i % window_size == 0) or (i == len(test_loader)-window_size):
            #if (i == 0) or (i == len(test_loader)-window_size):
            #if (i == 0) or (i / window_size == 10) or (i / window_size == 20) or (i / window_size == 30) or (i / window_size == 40) or (i == len(test_loader)-window_size):
            #if (i == 0) or (i / window_size == 10) or (i / window_size == 20) or (i / window_size == 30) or (i / window_size == 40):
            #if (i == 0) or (i / window_size == 5) or (i / window_size == 10) or (i / window_size == 15) or (i / window_size == 20) or (i / window_size == 25) or (i / window_size == 30) or (i / window_size == 35) or (i / window_size == 40):
            #if(i==0):
            #if(i >= 0) and (i <= 200):
            #if(i >= 0) and (i <= 200) or (i / window_size == 10) or (i / window_size == 20) or (i / window_size == 30) or (i / window_size == 40):
            #if(i==0) or (i / window_size == 10) or (i / window_size == 20) or (i / window_size == 30) or (i / window_size == 40):
            if(i >= 0) and (i <= 200) or(i >= window_size*10) and (i <= window_size*10+window_size) or(i >= window_size*20) and (i <= window_size*20+window_size) or(i >= window_size*30) and (i <= window_size*30+window_size) or(i >= window_size*40) and (i <= window_size*40+window_size):
                xy_pos = IMUTransformer(src, trg)
                #last_pos = xy_pos[:, -1, :]
                #xy_pos_pre = xy_pos
                xy_pos_pre = torch.cat([trg[:, 1:trg.shape[1], :], xy_pos[:, -1, :].unsqueeze(1)], dim=1)

                #xy_pos_pre = xy_pos
            else:
                #xy_pos = IMUTransformer(src, xy_pos)
                xy_pos = IMUTransformer(src, xy_pos_pre)
                xy_pos_pre = torch.cat([xy_pos_pre[:, 1:xy_pos_pre.shape[1], :], xy_pos[:, -1, :].unsqueeze(1)], dim=1)
                #xy_pos_pre = xy_pos

            #xy_pos_pre = IMUTransformer(src, trg)
            #xy_pos_pre = xy_pos
            loss = get_loss(xy_pos_pre, gt)
            #loss = get_loss(xy_pos, gt)
            print("My loss", '%.8f' % loss.item())

            if (i == 0):
                final_gt.append(gt)
                final_xy.append(xy_pos_pre)
            else:
                final_gt.append(gt[:, -1, :])
                final_xy.append(xy_pos_pre[:, -1, :])

            #result = '{:.4f} \t{:.4f} \t{:.4f}\t{:.4f} \t{:.4f}\t{:.4f}\n'.format()

        for j in range(len(final_gt)):
            final_gt[j] = final_gt[j].detach().cpu().numpy()
            final_xy[j] = final_xy[j].detach().cpu().numpy()

            if(j==0):
                np.savetxt(GTtxt, final_gt[j][0])
                np.savetxt(XYtxt, final_xy[j][0])
            else:
                np.savetxt(GTtxt, final_gt[j])
                #np.savetxt(GTtxt, final_gt[j])
                np.savetxt(XYtxt, final_xy[j])
                #np.savetxt(XYtxt, final_xy[j])
            '''
            if (j != len(final_gt) - 1):
                np.savetxt(GTtxt, final_gt[j][0])
                np.savetxt(GTtxt, final_gt[j][1])
                np.savetxt(XYtxt, final_xy[j][0])
                np.savetxt(XYtxt, final_xy[j][1])

            if(j==len(final_gt)-1):
                if(len(final_gt[j])==1):
                    np.savetxt(GTtxt, final_gt[j][0])
                    np.savetxt(XYtxt, final_xy[j][0])
                else:
                    np.savetxt(GTtxt, final_gt[j][0])
                    np.savetxt(GTtxt, final_gt[j][1])
                    np.savetxt(XYtxt, final_xy[j][0])
                    np.savetxt(XYtxt, final_xy[j][1])
            '''
    print("Test Finished Let's check the result!")

if __name__=='__main__':
    torch.cuda.empty_cache()

    print("IMU localization with Transformer")
    #train()
    test()