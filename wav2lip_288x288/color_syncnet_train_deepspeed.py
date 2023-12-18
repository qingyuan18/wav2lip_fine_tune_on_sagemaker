from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import time

from models import SyncNet_color as SyncNet
import audio

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob
import deepspeed

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
## Include DeepSpeed configuration arguments
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16

class Dataset(object):
    def __init__(self, split):
        self.all_videos = get_image_list(args.data_root, split)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))  # '{}.jpg' '{}.png'
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx]

            img_names = list(glob(join(vidname, '*.jpg')))  # '*.jpg' '*.png'
            # print('vidname:', vidname, 'img_names:', len(img_names))
            if len(img_names) <= 3 * syncnet_T:
                # print('CONTINUE: len(img_names) <= 3 * syncnet_T')
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)

            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                # print('CONTINUE: window_fnames is None. chosen:', chosen)
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            try:
                wavpath = join(vidname, "audio.wav")  # "audio.wav" "../audio.wav"
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                # print('CONTINUE Exception:', e, ', wavpath:', wavpath, ', wav:', wav)
                continue

            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                # print('CONTINUE: mel.shape[0] != syncnet_mel_step_size')
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]

            x = torch.FloatTensor(x)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            
            # print('x, mel, y:', x, mel, y)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    # print('a:', a.shape, ', v:', v.shape, ', y:', y.shape)
    # print('a:', torch.isnan(a).any(), ', v:', torch.isnan(v).any(), ', y:', torch.isnan(y).any())
    # print('a:', torch.isinf(a).any(), ', v:', torch.isinf(v).any(), ', y:', torch.isinf(y).any())
    d = nn.functional.cosine_similarity(a, v)
    # print('d:', d.shape, torch.isnan(d).any(), d)
    # print('y:', y.shape, torch.isnan(y).any(), y)
    d = torch.clamp(d, min=0.0)  # TODO Pytorch.clamp：将小于0的元素修改为0，截断元素的取值空间
    loss = logloss(d.unsqueeze(1), y)
    # print('loss:', loss)

    return loss

def train(device, model_engine, train_data_loader, test_data_loader, optimizer_deepspeed,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        prog_bar = tqdm(enumerate(train_data_loader))
        for step, (x, mel, y) in prog_bar:
        # for step, (x, mel, y) in enumerate(train_data_loader):
            model_engine.train()
            #optimizer.zero_grad()

            # Transform data to CUDA device
            # print('x:', x.shape)
            x = x.to(device)

            mel = mel.to(device)

            a, v = model_engine(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            model_engine.backward(loss)
            model_engine.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()

            if global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model_engine, optimizer_deepspeed, global_step, checkpoint_dir, global_epoch)

            #if global_step % hparams.syncnet_eval_interval == 0:
            #    with torch.no_grad():
            #        eval_model(test_data_loader, global_step, device, model, checkpoint_dir)

            prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
            # print('Loss: {}'.format(running_loss / (step + 1)))

        global_epoch += 1

def eval_model(test_data_loader, global_step, device, model, checkpoint_dir):
    eval_steps = 1400
    print('Evaluating for {} steps'.format(eval_steps))
    losses = []
    while 1:
        for step, (x, mel, y) in enumerate(test_data_loader):

            model.eval()

            # Transform data to CUDA device
            x = x.to(device)

            mel = mel.to(device)

            a, v = model(mel, x)
            y = y.to(device)

            loss = cosine_loss(a, v, y)
            losses.append(loss.item())

            if step > eval_steps: break

        averaged_loss = sum(losses) / len(losses)
        print(averaged_loss)

        return

def save_checkpoint(model_engine, optimizer, step, checkpoint_dir, epoch):
    # 获取当前时间戳
    timestamp = int(time.time())
    model_engine.save_checkpoint(checkpoint_dir)
    #torch_model = model_engine.module
    #output_model_path = checkpoint_dir+str(timestamp)+"_"+str(step)+"_trained_syncnet.pth"
    #torch.save(torch_model.state_dict(), output_model_path)
    print("Saved checkpoint:", checkpoint_dir)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=hparams.num_workers)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    ###deepspeed 改造#################
    model_engine, optimizer_deepspeed, _, _ = deepspeed.initialize(args=args,model=model,optimizer=optimizer)
    
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model_engine, train_data_loader, test_data_loader, optimizer_deepspeed,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)