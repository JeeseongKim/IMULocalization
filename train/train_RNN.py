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
from model.DenoiseRNN import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom

def train():
    #window_size = 200  # 200
    src_win_sz = 200
    tgt_win_sz = 2
    hidden_dim = 256

    num_epochs = 500
    batch_size = 256
    lr_drop = 300
    learning_rate = 1e-4
    weight_decay = 1e-5

    dtype = torch.FloatTensor

    vis = visdom.Visdom()
    plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))

    start = time.time()
    print("Training")
    #model definition
    IMUTransformer = IMU_Transformer_noisy(window_sz=src_win_sz, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.2).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()
    optimizer_IMU = torch.optim.AdamW(IMUTransformer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler_optimizer1 = torch.optim.lr_scheduler.StepLR(optimizer_IMU, lr_drop)

    DenoiseRNN = Denoise_RNN(inp_sz=256, hidden_sz=2, num_layers=4, dropout=0.1, bidirectional=True)
    DenoiseRNN = nn.DataParallel(DenoiseRNN).cuda()
    optimizer_Denoise = torch.optim.AdamW(DenoiseRNN.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler_optimizer2 = torch.optim.lr_scheduler.StepLR(optimizer_Denoise, lr_drop)

    #load ckpt
    load_ckpt = "/home/jsk/IMUlocalization/ckpt/train_model.pth"
    if os.path.exists(load_ckpt):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load(load_ckpt)
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])
        DenoiseRNN.module.load_state_dict(checkpoint['DenoiseRNN'])
        optimizer_IMU.load_state_dict(checkpoint['optimizer_IMU'])
        optimizer_Denoise.load_state_dict(checkpoint['optimizer_Denoise'])
        lr_scheduler_optimizer1.load_state_dict(checkpoint['lr_scheduler_optimizer1'])
        lr_scheduler_optimizer2.load_state_dict(checkpoint['lr_scheduler_optimizer2'])

    dataset = my_dataset_gyroz(window_size=src_win_sz)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    get_loss = pos_loss()

    #SaveLossTxt = open("MyLoss.txt", 'w')

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            cur_imu, cur_trg = data #(b,10,3)

            #src = cur_imu.cuda()*9.8 #m/s^2
            src = cur_imu.cuda() #m/s^2
            trg = cur_trg[:, cur_trg.shape[1]-1-tgt_win_sz:cur_trg.shape[1]-1, :].cuda() * 0.001 #mm -> m
            gt = cur_trg[:, cur_trg.shape[1]-tgt_win_sz:cur_trg.shape[1], :].cuda() * 0.001 #mm -> m

            noisy_xy = IMUTransformer(src, trg) #(2,256)
            p_noisy_xy = noisy_xy.permute(1, 0, 2)
            xy_pos = DenoiseRNN(p_noisy_xy)

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
        }, "/home/jsk/IMUlocalization/ckpt/train_10_square_dropout0.3/" + "js_ans2_" + str(epoch) + ".pth")

        vis.line(Y=[running_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plot_all, update='append')
        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))

def test():
    print("Testing")
    src_win_sz = 200
    tgt_win_sz = 2

    GTtxt = open("GT.txt", 'w')
    XYtxt = open("XY.txt", 'w')

    #model definition
    IMUTransformer = IMU_Transformer(window_sz=src_win_sz, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0).cuda()
    #IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=False).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()

    #MyCKPT = "/home/jsk/IMUlocalization/ckpt/train_10_square_dropout0.2/js_ans2_64.pth"
    #MyCKPT = "/home/jsk/IMUlocalization/ckpt/train_10_square_dropout0.2/js_ans2_143.pth"
    MyCKPT = "/home/jsk/IMUlocalization/ckpt/train_10_square_dropout0.3/js_ans2_64.pth"
    #MyCKPT = "/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth"
    #MyCKPT = "/home/jsk/IMUlocalization/ckpt/train_5_test_1_square/js_ans2_40.pth"
    #checkpoint save
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_5_test_1_square/js_ans2_100.pth"):
    #if os.path.exists("/home/jsk/IMUlocalization/ckpt/train_10_square/js_ans2_85.pth"):
    if os.path.exists(MyCKPT):
        print("-----Loading Checkpoint-----")
        print(MyCKPT)
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_5_test_1.pth")
        #checkpoint = torch.load("/home/jsk/IMUlocalization/ckpt/train_5_test_1_square/js_ans2_100.pth")
        checkpoint = torch.load(MyCKPT)
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])

    #imu_file = "/home/jsk/IMUlocalization/data/test/IMU/imu_1cycle_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data/test/Vicon/vicon_1cycle_01.txt"

    #imu_file = "/home/jsk/IMUlocalization/data_5_1/test/IMU/imu_1cycle_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data_5_1/test/Vicon/vicon_1cycle_01.txt"

    #imu_file = "/home/jsk/IMUlocalization/triangle_data/test/IMU/imu_tri_2cycle_04.txt"
    #vicon_file = "/home/jsk/IMUlocalization/triangle_data/test/Vicon/vicon_tri_2cycle_04.txt"

    #imu_file = "/home/jsk/IMUlocalization/pattern_data/test/IMU/imu_pattern_02.txt"
    #vicon_file = "/home/jsk/IMUlocalization/pattern_data/test/Vicon/vicon_pattern_02.txt"

    imu_file = "/home/jsk/IMUlocalization/data_1_5/test/IMU/imu_5cycle_01.txt"
    vicon_file = "/home/jsk/IMUlocalization/data_1_5/test/Vicon/vicon_5cycle_01.txt"

    #dataset = my_test_dataset(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    #dataset = my_test_dataset_modified(imu_file=imu_file, vicon_file=vicon_file, window_size=window_size)
    dataset = my_test_dataset_gyroz(imu_file=imu_file, vicon_file=vicon_file, window_size=src_win_sz)
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

            #src = cur_imu.cuda()
            #trg = cur_trg[:, 0:cur_trg.shape[1] - 1, :].cuda() * 0.001  # mm -> m (1~199)
            #gt = cur_trg[:, 1:cur_trg.shape[1], :].cuda() * 0.001  # mm -> m (2~200)

            src = cur_imu.cuda()  # m/s^2
            trg = cur_trg[:, cur_trg.shape[1] - 1 - tgt_win_sz:cur_trg.shape[1] - 1, :].cuda() * 0.001  # mm -> m
            gt = cur_trg[:, cur_trg.shape[1] - tgt_win_sz:cur_trg.shape[1], :].cuda() * 0.001  # mm -> m

            #xy_pos = IMUTransformer(src, trg)
            #xy_pos_pre = torch.cat([trg[:, 1:trg.shape[1], :], xy_pos[:, -1, :].unsqueeze(1)], dim=1)

            if(i==0):
                xy_pos = IMUTransformer(src, trg)
                #xy_pos_pre = xy_pos
                xy_pos_pre = torch.cat([trg[:, 1:trg.shape[1], :], xy_pos[:, -1, :].unsqueeze(1)], dim=1)

            else:
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

    print("Test Finished Let's check the result!")

if __name__=='__main__':
    torch.cuda.empty_cache()

    print("IMU localization with Transformer")
    #train()
    test()