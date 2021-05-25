import numpy as np
#from tqdm import tqdm
from utils import *
from loss import *
import warnings
import os
warnings.filterwarnings("ignore")
# from IQA_pytorch import SSIM
from torch.utils.data import TensorDataset, DataLoader
import time

from model.IMUTransformer import *

torch.multiprocessing.set_start_method('spawn', force=True)

import visdom
vis = visdom.Visdom()
plot_all = vis.line(Y=torch.tensor([0]), X=torch.tensor([0]), opts=dict(title='All Loss'))

window_size = 200 #200
hidden_dim = 256

num_epochs = 100
batch_size = 8

learning_rate = 1e-4
weight_decay = 1e-5

dtype = torch.FloatTensor

def train():
    start = time.time()

    #model definition
    #IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    #IMUTransformer = IMU_Transformer(ntoken=window_size, ninp=256, nhid=256, nhead=8, nlayers=6, dropout=0.5).cuda()
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()
    optimizer_IMU = torch.optim.AdamW(IMUTransformer.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #checkpoint save
    if os.path.exists("/home/jsk/IMUlocalization/train_model.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/IMUlocalization/train_model.pth")
        IMUTransformer.load_state_dict(checkpoint['IMUTransformer'])
        optimizer_IMU.load_state_dict(checkpoint['optimizer_IMU'])

    dataset = my_dataset(window_size=window_size)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    #for epoch in tqdm(range(num_epochs)):
    for epoch in (range(num_epochs)):
        running_loss = 0
        #for i, data in enumerate(tqdm(train_loader)):
        for i, data in enumerate((train_loader)):
            cur_imu, cur_vicon = data #(b,10,3)

            src = cur_imu.cuda() * 9.8 #m/s^2
            gt = cur_vicon.cuda() * 0.001 #mm -> m

            xy_pos = IMUTransformer(src)

            get_loss = pos_loss()
            loss = 0.3 * get_loss(xy_pos, gt)

            print("My loss", '%.8f' % loss.item())

            optimizer_IMU.zero_grad()

            loss.backward()

            optimizer_IMU.step()

            running_loss = running_loss + loss

        torch.save({
            'IMUTransformer': IMUTransformer.state_dict(),
            'optimizer_IMU': optimizer_IMU.state_dict(),
        }, "/home/jsk/IMUlocalization/train_model.pth")

        vis.line(Y=[running_loss], X=np.array([epoch]), win=plot_all, update='append')
        print('epoch [{}/{}], loss:{:.4f} '.format(epoch + 1, num_epochs, running_loss))


def test():
    start = time.time()

    #model definition
    IMUTransformer = IMU_Transformer(window_sz=window_size, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1).cuda()

    #checkpoint save
    if os.path.exists("/home/jsk/IMUlocalization/train_model.pth"):
        print("-----Loading Checkpoint-----")
        checkpoint = torch.load("/home/jsk/IMUlocalization/train_model.pth")
        IMUTransformer.load_state_dict(checkpoint['IMUTransformer'])

    dataset = my_test_dataset(window_size=window_size)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test_loss = 0
    #for i, data in enumerate(tqdm(train_loader)):
    for i, data in enumerate((test_loader)):
        cur_imu, cur_vicon = data #(b,10,3)

        src = cur_imu.cuda() * 9.8 #m/s^2
        gt = cur_vicon.cuda() * 0.001 #mm -> m

        xy_pos = IMUTransformer(src)

        get_loss = pos_loss()
        loss = get_loss(xy_pos, gt)

        print("My loss", '%.8f' % loss.item())
        test_loss = test_loss + loss







if __name__=='__main__':
    torch.cuda.empty_cache()

    print("IMU localization with Transformer")
    train()
    #test()