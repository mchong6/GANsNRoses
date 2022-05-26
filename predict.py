""" GANsNRoses: Selfie to Anime https://github.com/mchong6/GANsNRoses"""
import os
import tempfile
from base64 import b64encode

import cv2
import dlib
import kornia.augmentation as K
import moviepy.video.io.ImageSequenceClip
import numpy as np
import scipy
import torch
from aubio import source, tempo
from cog import BasePredictor, File, Input, Path
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

from model import *
from util import *

torch.backends.cudnn.benchmark = True


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(
        self,
        inpath: Path = Input(description="Input image or short video", default=None),
    ) -> Path:

        # get input file
        inpath = str(inpath)
       
        # model setup
        latent_dim = 8
        n_mlp = 5
        num_down = 3

        G_A2B = (
            Generator(
                256, 4, latent_dim, n_mlp, channel_multiplier=1, lr_mlp=0.01, n_res=1
            )
            .to(self.device)
            .eval()
        )
        ckpt = torch.load("GNR_checkpoint.pt", map_location=self.device)
        G_A2B.load_state_dict(ckpt["G_A2B_ema"])

        test_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True
                ),
            ]
        )

        if "mp4" in inpath:  # video
            print(f"*** Processing video input: {inpath} ***")

            # use normal mode for demo purposes (see original repo for other modes)
            mode = "normal"

            # Frame numbers and length of output video
            start_frame = 0
            end_frame = None
            frame_num = 0
            mp4_fps = 30
            faces = None
            smoothing_sec = 0.7
            eig_dir_idx = 1  # first eig isnt good so we skip it

            frames = []
            reader = cv2.VideoCapture(inpath)
            num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

            all_latents = torch.randn([8, latent_dim]).to(self.device)
            in_latent = all_latents

            # Face detector
            face_detector = dlib.get_frontal_face_detector()

            assert start_frame < num_frames - 1
            end_frame = end_frame if end_frame else num_frames

            while reader.isOpened():
                _, image = reader.read()
                if image is None:
                    break

                if frame_num < start_frame:
                    continue
                # Image size
                height, width = image.shape[:2]

                # 2. Detect with dlib
                if faces is None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_detector(gray, 1)
                if len(faces):
                    # For now only take biggest face
                    face = faces[0]

                # --- Prediction ---------------------------------------------------
                # Face crop with dlib and bounding box scale enlargement
                x, y, size = get_boundingbox(face, width, height)
                cropped_face = image[y : y + size, x : x + size]
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face = Image.fromarray(cropped_face)
                frame = test_transform(cropped_face).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    A2B_content, A2B_style = G_A2B.encode(frame)

                    in_latent = all_latents

                    fake_A2B = G_A2B.decode(A2B_content.repeat(8, 1, 1, 1), in_latent)

                    fake_A2B = torch.cat([fake_A2B[:4], frame, fake_A2B[4:]], 0)

                    fake_A2B = utils.make_grid(
                        fake_A2B.cpu(), normalize=True, range=(-1, 1), nrow=3
                    )

                # concatenate original image top
                fake_A2B = fake_A2B.permute(1, 2, 0).cpu().numpy()
                frames.append(fake_A2B * 255)

                frame_num += 1

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                frames, fps=mp4_fps
            )

            # save to temporary file. hack to make sure ffmpeg works
            output_path = Path(tempfile.mkdtemp()) / "output.mp4"
            clip.write_videofile(str(output_path))
            print(f'saving to {output_path}')

            return output_path

        # else, just process the image
        print(f"*** Processing image input: {inpath} ***")
        num_styles = 5
        style = torch.randn([num_styles, latent_dim]).to(self.device)

        # read input image
        image = cv2.imread(inpath)
        height, width = image.shape[:2]

        # Detect with dlib
        face_detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # grab first face
        face = face_detector(gray, 1)[0]

        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y : y + size, x : x + size]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        cropped_face = Image.fromarray(cropped_face)

        real_A = cropped_face
        real_A = test_transform(real_A).unsqueeze(0).to(self.device)

        with torch.no_grad():
            A2B_content, _ = G_A2B.encode(real_A)
            fake_A2B = G_A2B.decode(A2B_content.repeat(num_styles, 1, 1, 1), style)
            A2B = torch.cat([real_A, fake_A2B], 0)

        # create and save output
        output = utils.make_grid(A2B.cpu(), normalize=True, range=(-1, 1), nrow=10)
        output_path = Path(tempfile.mkdtemp()) / "output.png"
        torchvision.utils.save_image(output, output_path)
        print(f'saving to {output_path}')

        return output_path
