from os.path import dirname, join, basename, isfile
from tqdm import tqdm
from time import time

from models import SyncNet_color as SyncNet
import audio
import deepspeed
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list
import warnings
from torch.utils.data.distributed import DistributedSampler

# 忽略所有警告
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=True)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=True, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)
# -------------------------------- multi gpus --------------------------------------------
parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
# -------------------------------- multi gpus --------------------------------------------
args = parser.parse_args()

global_step = 0
global_epoch = 0
best_loss = 1000
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
			# frame = join(vidname, '{}.jpg'.format(frame_id))
			frame = join(vidname, f'{frame_id:05}.jpg')
			if not isfile(frame):
				return None
			window_fnames.append(frame)
		return window_fnames

	def crop_audio_window(self, spec, start_frame):
		# num_frames = (T x hop_size * fps) / sample_rate
		start_frame_num = self.get_frame_id(start_frame)
		start_idx = int(80. * (start_frame_num / float(hparams.fps)))

		end_idx = start_idx + syncnet_mel_step_size

		return spec[start_idx: end_idx, :]

	def __len__(self):
		return len(self.all_videos)

	def __getitem__(self, idx):
		while 1:
			idx = random.randint(0, len(self.all_videos) - 1)
			vidname = self.all_videos[idx]
			img_names = list(glob(join(vidname, '*.jpg')))
			if len(img_names) <= 3 * syncnet_T:
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
				wavpath = join(vidname, "audio.wav")
				wav = audio.load_wav(wavpath, hparams.sample_rate)

				orig_mel = audio.melspectrogram(wav).T
			except Exception as e:
				continue

			mel = self.crop_audio_window(orig_mel.copy(), img_name)

			if (mel.shape[0] != syncnet_mel_step_size):
				continue

			# H x W x 3 * T
			x = np.concatenate(window, axis=2) / 255.
			x = x.transpose(2, 0, 1)
			x = x[:, x.shape[1] // 2:]

			x = torch.FloatTensor(x)
			mel = torch.FloatTensor(mel.T).unsqueeze(0)
			return x, mel, y


def cosine_loss(a, v, y):
	logloss = nn.BCELoss()
	d = nn.functional.cosine_similarity(a, v)
	loss = logloss(d.unsqueeze(1), y)
	return loss


def train(device, model_engine, train_data_loader, test_data_loader, optimizer_deepspeed,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, sampler=None):
	global global_step, global_epoch

	while global_epoch < nepochs:
		sampler.set_epoch(global_epoch)
		st_e = time()
		running_loss = 0.
		for step, (x, mel, y) in enumerate(train_data_loader):
			st = time()
			model_engine.train()

			# Transform data to CUDA device
			x = x.to(device)
			mel = mel.to(device)
			a, v = model_engine(mel, x)
			y = y.to(device)

			loss = cosine_loss(a, v, y)
			model_engine.backward(loss)
			model_engine.step()

			global_step += 1
			running_loss += loss.item()

			if global_step == 1 or global_step % checkpoint_interval == 0:
				saved_model_name = join(checkpoint_dir, f"step{global_step:09d}_{running_loss / (step + 1):.8f}.pth")
				save_checkpoint(model_engine, optimizer_deepspeed, global_step, saved_model_name, global_epoch)

			#if global_step % hparams.syncnet_eval_interval == 0:
			#	with torch.no_grad():
			#		eval_loss = eval_model(test_data_loader, device, model_engine)
			#		if eval_loss < best_loss:
			#			saved_model_name = join(checkpoint_dir, f"best.pth")
			#			save_checkpoint(model_engine, optimizer_deepspeed, global_step, saved_model_name, global_epoch)
			#			best_loss = eval_loss
			# prog_bar.set_description('Loss: {}'.format(running_loss / (step + 1)))
			print(f"Step {global_step} | Loss: {running_loss / (step + 1):.8f} | Elapsed: {(time() - st):.5f}")

		global_epoch += 1


def eval_model(test_data_loader, device, model):
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

			if step > eval_steps:
				break

		averaged_loss = sum(losses) / len(losses)
		print(f"eval loss: {averaged_loss}")

		return averaged_loss


# def save_checkpoint(model_engine, optimizer, step, checkpoint_dir, epoch):
# 	# 获取当前时间戳
# 	model_engine.save_checkpoint(checkpoint_dir)
# 	# torch_model = model_engine.module
# 	# output_model_path = checkpoint_dir+str(timestamp)+"_"+str(step)+"_trained_syncnet.pth"
# 	# torch.save(torch_model.state_dict(), output_model_path)
# 	print("Saved checkpoint:", checkpoint_dir)


def save_checkpoint(model_engine, optimizer, step, checkpoint_path, epoch):
	# checkpoint_path = join(	checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
	optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
	torch.save({
		"state_dict": model_engine.state_dict(),
		"optimizer": optimizer_state,
		"global_step": step,
		"global_epoch": epoch,
		"best_loss": best_loss,
	}, checkpoint_path)
	print("Saved checkpoint:", checkpoint_path)


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

	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)


	device = torch.device("cuda" if use_cuda else "cpu")

	# Model
	model = SyncNet().to(device)
	print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hparams.syncnet_lr)

	# deepspeed 改造
	model_engine, optimizer_deepspeed, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)
    
    # Dataset and Dataloader setup
	train_dataset = Dataset('train')
	test_dataset = Dataset('val')
	sampler = data_utils.DistributedSampler(train_dataset, shuffle=True)
	train_data_loader = data_utils.DataLoader(
		train_dataset, batch_size=hparams.syncnet_batch_size, 
		num_workers=hparams.num_workers, sampler=sampler)

	test_data_loader = data_utils.DataLoader(
		test_dataset, batch_size=hparams.syncnet_batch_size,
		num_workers=hparams.num_workers,sampler=sampler)

	print('train_dataset:', len(train_dataset))
	print('test_dataset:', len(test_dataset))
    
    
	if checkpoint_path is not None:
		load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

	train(
		device, model_engine, train_data_loader, test_data_loader, optimizer_deepspeed,
		checkpoint_dir=checkpoint_dir,
		checkpoint_interval=hparams.syncnet_checkpoint_interval,
		nepochs=hparams.nepochs,
		sampler=sampler)
