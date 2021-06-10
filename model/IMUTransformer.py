import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import resnet50
#from torch.nn import TransformerEncoder, TransformerEncoderLayer
#import torch.nn.TransformerEncoder as TransformerEncoder
#import torch.nn.TrasnformerEncoderLayer as TransformerEncoderLayer
from model.Transformer import *
'''
class IMU_Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=256, dropout=0.1):
        super(IMU_Transformer, self).__init__()

        self.Transformer = torch.nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_4 = torch.nn.Conv1d(128, d_model, kernel_size=3, padding=1, dilation=1, stride=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, trg):
        p_src = src.permute(0, 2, 1)
        src_1 = self.conv_1(p_src)
        src_2 = self.conv_2(src_1)
        src_3 = self.conv_3(src_2)
        src_4 = self.conv_4(src_3)

        src_mask = self.generate_square_subsequent_mask(src_4.shape[2]).cuda()

        h = self.Transformer(src=src_4, tgt=, src_mask=src_mask, )
        return output
'''

'''
class IMU_Transformer(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(IMU_Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

        self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_4 = torch.nn.Conv1d(128, nhid, kernel_size=3, padding=1, dilation=1, stride=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        p_src = src.permute(0, 2, 1)
        src_1 = self.conv_1(p_src)
        src_2 = self.conv_2(src_1)
        src_3 = self.conv_3(src_2)
        src_4 = self.conv_4(src_3)

        src_mask = self.generate_square_subsequent_mask(src_4.shape[2]).cuda()

        encoder_inp = self.encoder(src_4.long()) * math.sqrt(self.ninp)
        enc_inp = self.pos_encoder(encoder_inp)
        output = self.transformer_encoder(enc_inp, src_mask)
        output = self.decoder(output)
        return output
'''


class IMU_Transformer_noisy(nn.Module):
    def __init__(self, window_sz, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(IMU_Transformer, self).__init__()

        # self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
        self.transformer = Transformer(256, nheads, num_encoder_layers, num_decoder_layers, hidden_dim, dropout)
        self.get_pos = MLP(hidden_dim, hidden_dim, 2, 3)
        # self.get_pos_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.get_pos_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.get_pos_3 = torch.nn.Linear(hidden_dim, 2)
        self.pos_encoder = PositionalEncoding(window_sz, dropout)
        # self.pos_decoder = PositionalEncoding(2, dropout)

        self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        # self.conv_1 = torch.nn.Conv1d(6, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.cconv_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_4 = torch.nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1, dilation=1, stride=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, gt):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # src = (1,200,3)
        p_src = src.permute(0, 2, 1)
        src_1 = self.conv_1(p_src)
        src_2 = self.conv_2(src_1)
        src_3 = self.conv_3(src_2)
        src_4 = self.conv_4(src_3)

        p_gt = gt.permute(0, 2, 1)
        gt_1 = self.cconv_1(p_gt)
        gt_2 = self.conv_2(gt_1)
        gt_3 = self.conv_3(gt_2)
        gt_4 = self.conv_4(gt_3)

        enc_inp = src_4.permute(2, 0, 1)
        src_mask = self.generate_square_subsequent_mask(enc_inp.shape[0]).cuda()
        trg_mask = self.generate_square_subsequent_mask(gt_4.shape[2]).cuda()
        # tgt = self.query_embed.weight.unsqueeze(1).repeat(1, enc_inp.shape[1], 1)
        tgt = gt_4.permute(2, 0, 1)

        pos = self.pos_encoder(src_4.cuda()).permute(2, 0, 1)
        encoder_input = pos + 0.1 * enc_inp
        # encoder_input = enc_inp

        # tgt_pos = self.pos_decoder(gt_4.cuda()).permute(2, 0, 1)
        # decoder_input = tgt_pos + 0.1 * tgt

        h_kp = self.transformer(src=encoder_input, tgt=tgt, src_mask=src_mask, tgt_mask=trg_mask)
        # h_kp = self.transformer(src=encoder_input, tgt=decoder_input, src_mask=src_mask, tgt_mask=trg_mask)

        XY_pos = self.get_pos(h_kp)  ##

        ans = XY_pos.permute(1, 0, 2)
        return ans


class IMU_Transformer(nn.Module):
        def __init__(self, window_sz, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
            super(IMU_Transformer, self).__init__()

            #self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
            self.transformer = Transformer(256, nheads, num_encoder_layers, num_decoder_layers, hidden_dim, dropout)
            self.get_pos = MLP(hidden_dim, hidden_dim, 2, 3)
            #self.get_pos_1 = torch.nn.Linear(hidden_dim, hidden_dim)
            #self.get_pos_2 = torch.nn.Linear(hidden_dim, hidden_dim)
            #self.get_pos_3 = torch.nn.Linear(hidden_dim, 2)
            self.pos_encoder = PositionalEncoding(window_sz, dropout)
            #self.pos_decoder = PositionalEncoding(2, dropout)

            self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
            #self.conv_1 = torch.nn.Conv1d(6, 16, kernel_size=3, padding=1, dilation=1, stride=1)
            self.cconv_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1, dilation=1, stride=1)
            self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
            self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
            self.conv_4 = torch.nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1, dilation=1, stride=1)

        def generate_square_subsequent_mask(self, sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        def forward(self, src, gt):
            # propagate inputs through ResNet-50 up to avg-pool layer
            #src = (1,200,3)
            p_src = src.permute(0, 2, 1)
            src_1 = self.conv_1(p_src)
            src_2 = self.conv_2(src_1)
            src_3 = self.conv_3(src_2)
            src_4 = self.conv_4(src_3)

            p_gt = gt.permute(0, 2, 1)
            gt_1 = self.cconv_1(p_gt)
            gt_2 = self.conv_2(gt_1)
            gt_3 = self.conv_3(gt_2)
            gt_4 = self.conv_4(gt_3)

            enc_inp = src_4.permute(2, 0, 1)
            src_mask = self.generate_square_subsequent_mask(enc_inp.shape[0]).cuda()
            trg_mask = self.generate_square_subsequent_mask(gt_4.shape[2]).cuda()
            #tgt = self.query_embed.weight.unsqueeze(1).repeat(1, enc_inp.shape[1], 1)
            tgt = gt_4.permute(2, 0, 1)

            pos = self.pos_encoder(src_4.cuda()).permute(2, 0, 1)
            encoder_input = pos + 0.1*enc_inp
            #encoder_input = enc_inp

            #tgt_pos = self.pos_decoder(gt_4.cuda()).permute(2, 0, 1)
            #decoder_input = tgt_pos + 0.1 * tgt

            h_kp = self.transformer(src=encoder_input, tgt=tgt, src_mask=src_mask, tgt_mask=trg_mask)
            #h_kp = self.transformer(src=encoder_input, tgt=decoder_input, src_mask=src_mask, tgt_mask=trg_mask)

            XY_pos = self.get_pos(h_kp) ##

            ans = XY_pos.permute(1, 0, 2)
            return ans


class IMU_Transformer_dist(nn.Module):
    def __init__(self, window_sz, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(IMU_Transformer_dist, self).__init__()

        # self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
        self.transformer = Transformer(256, nheads, num_encoder_layers, num_decoder_layers, hidden_dim, dropout)
        self.get_pos = MLP(hidden_dim, hidden_dim, 1, 3)
        # self.get_pos_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.get_pos_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.get_pos_3 = torch.nn.Linear(hidden_dim, 2)
        self.pos_encoder = PositionalEncoding(window_sz, dropout)

        self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        # self.conv_1 = torch.nn.Conv1d(6, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.cconv_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_4 = torch.nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1, dilation=1, stride=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, gt):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # src = (1,200,3)
        p_src = src.permute(0, 2, 1)
        src_1 = self.conv_1(p_src)
        src_2 = self.conv_2(src_1)
        src_3 = self.conv_3(src_2)
        src_4 = self.conv_4(src_3)

        p_gt = gt.permute(0, 2, 1)
        gt_1 = self.cconv_1(p_gt)
        gt_2 = self.conv_2(gt_1)
        gt_3 = self.conv_3(gt_2)
        gt_4 = self.conv_4(gt_3)

        enc_inp = src_4.permute(2, 0, 1)
        src_mask = self.generate_square_subsequent_mask(enc_inp.shape[0]).cuda()
        trg_mask = self.generate_square_subsequent_mask(gt_4.shape[2]).cuda()
        # tgt = self.query_embed.weight.unsqueeze(1).repeat(1, enc_inp.shape[1], 1)
        tgt = gt_4.permute(2, 0, 1)

        pos = self.pos_encoder(src_4.cuda()).permute(2, 0, 1)
        encoder_input = pos + 0.1 * enc_inp
        # encoder_input = enc_inp

        pos_tgt = self.pos_encoder(gt.cuda()).permute(2, 0, 1)
        tgt = pos_tgt + 0.1 * tgt
        h_kp = self.transformer(src=encoder_input, tgt=tgt, src_mask=src_mask, tgt_mask=trg_mask)

        # get_pos_1 = self.get_pos_1(h_kp.clone()) ##
        # get_pos_2 = self.get_pos_2(get_pos_1) ##
        # XY_pos = self.get_pos_3(get_pos_2) ##
        XY_pos = self.get_pos(h_kp)  ##

        ans = XY_pos.permute(1, 0, 2)
        return ans


class IMU_Transformer_inp3(nn.Module):
    def __init__(self, window_sz, hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dropout):
        super(IMU_Transformer_inp3, self).__init__()

        # self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, return_intermediate_dec=True)
        self.transformer = torch.nn.Transformer(256, nheads, num_encoder_layers, num_decoder_layers, hidden_dim,dropout)
        self.query_embed = nn.Embedding(window_sz, hidden_dim)
        self.get_pos = MLP(hidden_dim, hidden_dim, 2, 3)
        self.pos_encoder = PositionalEncoding(window_sz, dropout)

        self.conv_1 = torch.nn.Conv1d(3, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        #self.conv_1 = torch.nn.Conv1d(6, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.cconv_1 = torch.nn.Conv1d(2, 16, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_2 = torch.nn.Conv1d(16, 64, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_3 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1, dilation=1, stride=1)
        self.conv_4 = torch.nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1, dilation=1, stride=1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, gt):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # src = (1,200,3)
        p_src = src.permute(0, 2, 1)
        src_1 = self.conv_1(p_src)
        src_2 = self.conv_2(src_1)
        src_3 = self.conv_3(src_2)
        src_4 = self.conv_4(src_3)

        p_gt = gt.permute(0, 2, 1)
        gt_1 = self.cconv_1(p_gt)
        gt_2 = self.conv_2(gt_1)
        gt_3 = self.conv_3(gt_2)
        gt_4 = self.conv_4(gt_3)

        enc_inp = src_4.permute(2, 0, 1)
        src_mask = self.generate_square_subsequent_mask(enc_inp.shape[0]).cuda()
        trg_mask = self.generate_square_subsequent_mask(gt_4.shape[2]).cuda()
        # tgt = self.query_embed.weight.unsqueeze(1).repeat(1, enc_inp.shape[1], 1)
        tgt = gt_4.permute(2, 0, 1)

        pos = self.pos_encoder(src_4.cuda()).permute(2, 0, 1)
        encoder_input = pos + 0.1 * enc_inp
        # encoder_input = enc_inp

        h_kp = self.transformer(src=encoder_input, tgt=tgt, src_mask=src_mask, tgt_mask=trg_mask)

        XY_pos = self.get_pos(h_kp)

        return XY_pos.permute(1, 0, 2)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x.clone())) if i < self.num_layers - 1 else layer(x)
        return x.clone()