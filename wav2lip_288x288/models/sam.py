import torch
from torch import nn
from torch.nn import functional as F
import math
from torchsummary import summary

from .conv import Conv2dTranspose, Conv2d, nonorm_Conv2d


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SAM(nn.Module):
    def __init__(self):
        super(SAM, self).__init__()
        self.sa = SpatialAttention()

    def forward(self,sp,se):
        sp_att = self.sa(sp)
        out = se*sp_att+se
        return out


class Wav2Lip_SAM(nn.Module):
    def __init__(self, audio_encoder=None):
        super(Wav2Lip_SAM, self).__init__()
        self.sam = SAM()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True)), # 192, 192
            # nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)), # 192, 192

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 96, 96
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 48, 48
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 24, 24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 12, 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 6, 6
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),


            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),       # 3, 3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)),])
            ##################


        if audio_encoder is None:
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

                ###################
                # Modified blocks
                ##################
                Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
                Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
                    ##################
        else:
            self.audio_encoder = audio_encoder

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        self.audio_refine = nn.Sequential(
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),),  # + 1024

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),), # + 1024

            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6  + 512

            nn.Sequential(Conv2dTranspose(1536, 768, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12  + 256

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24  + 128
            ##################

            nn.Sequential(Conv2dTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48  + 64

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 96, 96  + 32

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 192, 192  + 16

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def new_refine(self):
        self.audio_refine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.Tanh())

    def forward(self, audio_sequences, face_sequences, noise=False):
        if noise:
            return self.forward_with_noise(audio_sequences, face_sequences)
        # audio_sequences = (B, T, 1, 80, 16)
        # B = audio_sequences.size(0)

        # input_dim_size = len(face_sequences.size())
        # if input_dim_size > 4:
        #     face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)


        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        audio_embedding = audio_embedding.detach()
        audio_embedding = self.audio_refine(audio_embedding)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        audio_embedding = torch.clamp(audio_embedding, -1, 1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = self.sam(feats[-1], x)
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

class Wav2Lip_384(nn.Module):
    def __init__(self, audio_encoder=None, is_sam=True):
        super(Wav2Lip_384, self).__init__()
        self.is_sam = is_sam
        self.sam = SAM()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 8, kernel_size=7, stride=1, padding=3)), # 384, 384
            
            nn.Sequential(Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True)), # 192, 192

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 96, 96
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 48, 48
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 24, 24
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),       # 12, 12
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),     # 6, 6
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),),

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),       # 3, 3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0),     # 1, 1
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)),])
            ##################


        if audio_encoder is None:
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

                ###################
                # Modified blocks
                ##################
                Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
                Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
                    ##################
        else:
            self.audio_encoder = audio_encoder

        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        self.audio_refine = nn.Sequential(
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),),  # + 1024

            ###################
            # Modified blocks
            ##################
            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=1, padding=0), # 3,3
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),), # + 1024

            nn.Sequential(Conv2dTranspose(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, residual=True),), # 6, 6  + 512

            nn.Sequential(Conv2dTranspose(1536, 768, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(768, 768, kernel_size=3, stride=1, padding=1, residual=True),), # 12, 12  + 256

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),), # 24, 24  + 128
            ##################

            nn.Sequential(Conv2dTranspose(640, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),), # 48, 48  + 64

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),), # 96, 96  + 32

            nn.Sequential(Conv2dTranspose(160, 80, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(80, 80, kernel_size=3, stride=1, padding=1, residual=True),), # 192, 192  + 16
            
            nn.Sequential(Conv2dTranspose(96, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),),]) # 384, 384  + 8

        self.output_block = nn.Sequential(Conv2d(72, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
    
    def forward(self, audio_sequences, face_sequences):


        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        audio_embedding = audio_embedding.detach()
        audio_embedding = self.audio_refine(audio_embedding)
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        audio_embedding = torch.clamp(audio_embedding, -1, 1)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        cnt = 0
        for f in self.face_decoder_blocks:
            print(f"cnt: {cnt}")
            x = f(x)
            try:
                if self.is_sam:
                    x = self.sam(feats[-1], x)
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e
#             print("x shape: ", x.shape)
            feats.pop()
            cnt += 1

        x = self.output_block(x)
        if input_dim_size > 4:
            x = torch.split(x, B, dim=0) # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2) # (B, C, T, H, W)

        else:
            outputs = x

        return outputs