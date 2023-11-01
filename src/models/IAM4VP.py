import math

import lightning as L
import torch
import torch.nn.functional as F
from src.models.modules import ConvSC, ConvNeXt_block, Attention, ConvNeXt_bottle
from torch import nn

import numpy as np


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class LP(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(LP, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(640, 64, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        Y = Y.reshape(int(ys[0] / 10), int(ys[1] * 10), 64, 64)
        Y = self.readout(Y)
        return Y


class Predictor(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T):
        super(Predictor, self).__init__()

        self.N_T = N_T
        st_block = [ConvNeXt_bottle(dim=channel_in)]
        for i in range(0, N_T):
            st_block.append(ConvNeXt_block(dim=channel_in))

        self.st_block = nn.Sequential(*st_block)

    def forward(self, x, time_emb):
        B, T, C, H, W = x.shape
        # (1, 1024, 63, 63)
        x = x.reshape(B, T * C, H, W)
        z = self.st_block[0](x, time_emb)
        for i in range(1, self.N_T):
            z = self.st_block[i](z, time_emb)

        y = z.reshape(B, int(T / 2), C, H, W)
        return y


class IAM4VP(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=512, N_S=4, N_T=6):
        super(IAM4VP, self).__init__()
        T, C, H, W = shape_in
        self.time_mlp = Time_MLP(dim=64)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Predictor(T * hid_S, hid_T, N_T)
        self.dec = Decoder(hid_S, C, N_S)
        self.attn = Attention(64)
        self.readout = nn.Conv2d(64, 1, 1)
        self.mask_token = nn.Parameter(torch.zeros(N_T, hid_S, 63, 63))
        self.lp = LP(C, hid_S, N_S)

    def forward(self, x_raw, y_raw=None, t=None):
        B, T, C, H, W = x_raw.shape
        _, Ty, _, _, _ = y_raw.shape

        x = x_raw.view(B * T, C, H, W)
        time_emb = self.time_mlp(t)
        embed, skip = self.enc(x)
        mask_token = self.mask_token.repeat(B, 1, 1, 1, 1)

        for idx, pred in enumerate(y_raw.view(B * Ty, C, H, W)):
            embed2, _ = self.lp(pred)
            mask_token[:, idx, :, :, :] = embed2

        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        z2 = mask_token
        z = torch.cat([z, z2], dim=1)
        # (batch, seq_in, 64, 63, 63) (1, 64)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = self.attn(Y)
        Y = self.readout(Y)

        return Y


class ImplicitStackedAutoregressiveForVideoPrediction(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.model = IAM4VP(
            # (T (seq_input + seq_output), channel, h, w)
            shape_in=(16, 1, 252, 252),
            hid_S=64,
            hid_T=512,
            N_S=4,
            N_T=12
        )

    def forward(self, x, y):
        x, t = x

        x = x.to(device=self.model.readout.weight.device)
        t = torch.tensor(np.array(t, dtype=np.int64)).repeat(x.shape[0])

        output = self.model(x_raw=x, y_raw=y, t=t)
        return output

    def training_step(self, batch):
        x, y = batch
        # x - shape (batch_size, seq_len, channel, h, w)
        out = self.forward(x, y)
        out[y == -1] = -1
        loss = F.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
