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

window_size = 200 #200
hidden_dim = 256

num_epochs = 10
batch_size = 128

learning_rate = 1e-4
weight_decay = 1e-5

dtype = torch.FloatTensor

def train():
    vis = visdom.Visdom()
    plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))

    start = time.time()
    print("Training")

    #model definition
    #IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    #IMUTransformer = IMU_Transformer(ntoken=window_size, ninp=256, nhid=256, nhead=8, nlayers=6, dropout=0.5).cuda()
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()
    optimizer_IMU = torch.optim.AdamW(IMUTransformer.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #checkpoint save
    if os.path.exists("/home/jsk/IMUlocalization/train_model.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/IMUlocalization/train_model.pth")
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])
        optimizer_IMU.load_state_dict(checkpoint['optimizer_IMU'])

    dataset = my_dataset(window_size=window_size)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for i, data in enumerate(tqdm(train_loader)):
            cur_imu, cur_vicon = data #(b,10,3)

            src = cur_imu.cuda() * 9.8 #m/s^2
            gt = cur_vicon.cuda() * 0.001 #mm -> m

            xy_pos = IMUTransformer(src, gt)

            get_loss = pos_loss()
            loss = get_loss(xy_pos, gt)

            print("My loss", '%.8f' % loss.item())

            optimizer_IMU.zero_grad()

            loss.backward()

            optimizer_IMU.step()

            running_loss = running_loss + loss

        torch.save({
            'IMUTransformer': IMUTransformer.module.state_dict(),
            'optimizer_IMU': optimizer_IMU.state_dict(),
        }, "/home/jsk/IMUlocalization/train_model.pth")

        vis.line(Y=[running_loss.detach().cpu().numpy()], X=np.array([epoch]), win=plot_all, update='append')
        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))


def test():
    print("Testing")
    start = time.time()

    GTtxt = open("GT.txt", 'w')
    XYtxt = open("XY.txt", 'w')

    #model definition
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    IMUTransformer = nn.DataParallel(IMUTransformer).cuda()

    #checkpoint save
    if os.path.exists("/home/jsk/IMUlocalization/train_model.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/IMUlocalization/train_model.pth")
        IMUTransformer.module.load_state_dict(checkpoint['IMUTransformer'])

    #imu_file = "/home/jsk/IMUlocalization/data/test/IMU/imu_1cycle_01.txt"
    #vicon_file = "/home/jsk/IMUlocalization/data/test/Vicon/vicon_1cycle_01.txt"

    imu_file = "/home/jsk/IMUlocalization/data_5_1/test/IMU/imu_1cycle_05.txt"
    vicon_file = "/home/jsk/IMUlocalization/data_5_1/test/Vicon/vicon_1cycle_05.txt"

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

        np.savetxt(GTtxt, final_gt[j][0])
        np.savetxt(GTtxt, final_gt[j][1])
        np.savetxt(XYtxt, final_xy[j][0])
        np.savetxt(XYtxt, final_xy[j][1])


if __name__=='__main__':
    torch.cuda.empty_cache()

    print("IMU localization with Transformer")
    train()
    #test()