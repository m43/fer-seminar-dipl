import glob
import os
import sys
from abc import ABC, abstractmethod

from tqdm import tqdm

sys.path.append('./RAFT')
sys.path.append('./RAFT/core')

import argparse
import cv2
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder, forward_interpolate

import pathlib


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class FrameSequence(ABC):
    @abstractmethod
    def nextFrame(self):
        pass

    @abstractmethod
    def totalFrames(self):
        pass

    def release(self):
        pass


class VideoFrameSequence(FrameSequence):
    def __init__(self, path):
        if not path:
            print("Using camera as input")
            self.cap = cv2.VideoCapture(0)
            self.n_frames = -1
            return

        self.n_frames = 0
        tmp_cap = cv2.VideoCapture(path)
        ret = True
        while ret:
            ret, frame = tmp_cap.read()
            self.n_frames += 1

        self.cap = cv2.VideoCapture(path)

    def totalFrames(self):
        return self.n_frames

    def nextFrame(self):
        ret, frame = self.cap.read()

        if not ret:
            return None

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr

    def release(self):
        self.cap.release()


class ImageFrameSequence(FrameSequence):
    """
    Supports only .jpg and .png images!
    """

    def __init__(self, path):
        self.images = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
        assert self.images
        self.i = 0

    def totalFrames(self):
        return len(self.images)

    def nextFrame(self):
        if self.i == len(self.images):
            return None
        im_path, self.i = self.images[self.i], self.i + 1
        img_bgr = np.array(Image.open(im_path)).astype(np.uint8)
        return img_bgr


def preprocess_image(img_bgr):
    img_torch = torch.from_numpy(img_bgr).permute(2, 0, 1).float()
    return img_torch[None].to(args.device)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

    model = model.module
    model.to(args.device)
    model.eval()

    ensure_dir(args.save_dir)
    if args.video:
        fs = VideoFrameSequence(args.path)
    else:
        fs = ImageFrameSequence(args.path)

    frame = fs.nextFrame()
    height, width, channels = frame.shape
    # print(f"Frame shape: {frame.shape}")
    image1 = preprocess_image(frame)
    print(f"Image shape: {image1.shape}")
    padder = InputPadder(image1.shape)
    image1 = padder.pad(image1)[0]

    video_img = cv2.VideoWriter(
        os.path.join(args.save_dir, '0_img.avi'), cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (width, height))
    video_flo = cv2.VideoWriter(
        os.path.join(args.save_dir, '0_flo.avi'), cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (width, height))
    video_imgflo = cv2.VideoWriter(
        os.path.join(args.save_dir, '0_imgflo.avi'), cv2.VideoWriter_fourcc(*'MJPG'), args.fps, (width, height * 2))

    frames_to_process = fs.totalFrames()
    if args.max_frames != -1:
        frames_to_process = min(args.max_frames, frames_to_process)
    flow_prev = None
    for i in tqdm(range(1, (4000000000 if frames_to_process == -1 else frames_to_process) + 1)):
        frame = fs.nextFrame()
        if frame is None:
            break

        image2 = preprocess_image(frame)
        image2 = padder.pad(image2)[0]
        with torch.no_grad():
            flow_low, flow_up = model(image1, image2, iters=args.iters, test_mode=True, flow_init=flow_prev)
        if args.warm_start:
            flow_prev = forward_interpolate(flow_low[0])[None]

        img = padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy().astype(np.uint8)[:, :, [2, 1, 0]]
        flo = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flo = flow_viz.flow_to_image(flo).astype(np.uint8)[:, :, [2, 1, 0]]
        # flo = img
        img_flo = np.concatenate([img, flo], axis=0)

        if args.log_images:
            cv2.imwrite(os.path.join(args.save_dir, f'1_img_{i:05d}.png'), img)
            cv2.imwrite(os.path.join(args.save_dir, f'2_flo_{i:05d}.png'), flo)
            cv2.imwrite(os.path.join(args.save_dir, f'3_imgflo_{i:05d}.png'), img_flo)

        video_img.write(img)
        video_flo.write(flo)
        video_imgflo.write(img_flo)

        image1, image2 = image2, None

    fs.release()
    video_img.release()
    video_flo.release()
    video_imgflo.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--video', type=int)
    parser.add_argument('--log_images', type=int, default=0)
    parser.add_argument('--warm_start', type=int, default=0)
    parser.add_argument('--save_dir', default="./imgs/demo")
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=-1, help='Upper bound for number of frames to process')
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')
    args = parser.parse_args()

    print(args)
    demo(args)

# Examples of runs:
# python -m playground.video_to_flow --device cpu --model RAFT/models/raft-kitti.pth --video 1 --warm_start 1 --save_dir "./imgs/demo2"
# python -m playground.video_to_flow --fps 30 --device cuda --model RAFT/models/raft-kitti.pth --video 1 --path "./data/ulaz u crnu riku.mp4" --warm_start 1 --save_dir "./imgs/rika-kitti-warm"
# python -m playground.video_to_flow --fps 30 --max_frames="3000" --device cuda --model RAFT/models/raft-kitti.pth --video 1 --path "./data/ladja.mp4" --warm_start 1 --save_dir "./imgs/ladja-kitti-warm"
# python -m playground.video_to_flow --fps 6 --device cpu --model RAFT/models/raft-kitti.pth --video 0 --path "/mnt/terra/data/Sintel/training/final/ambush_6" --save_dir "./imgs/sintel-kittimodel/training-final-ambush_6"
