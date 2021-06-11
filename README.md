# GANs N' Roses Pytorch
![](teaser.pdf)

This is the PyTorch implementation of GANs N’ Roses: Stable, Controllable, Diverse Image to Image Translation (works for videos too!).

>**Abstract:**<br>
>We show how to learn a map that takes a content code, derived from a face image, and a randomly chosen style code to an anime image. We derive an adversarial loss from our simple and effective definitions of style and content. This adversarial loss guarantees the map is diverse -- a very wide range of anime can be produced from a single content code. Under plausible assumptions, the map is not just diverse, but also correctly represents the probability of an anime, conditioned on an input face. In contrast, current multimodal generation procedures cannot capture the complex styles that appear in anime.  Extensive quantitative experiments support the idea the map is correct. Extensive qualitative results show that the method can generate a much more diverse range of styles than SOTA comparisons. Finally, we show that our formalization of content and style allows us to perform video to video translation without ever training on videos.

## Dependency
Our codebase is based off [stylegan2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch). 
```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install tqdm gdown kornia scipy opencv-python dlib moviepy lpips
```

## Dataset
```
└── YOUR_DATASET_NAME
   ├── trainA
       ├── xxx.jpg (name, format doesn't matter)
       ├── yyy.png
       └── ...
   ├── trainB
       ├── zzz.jpg
       ├── www.png
       └── ...
   ├── testA
       ├── aaa.jpg 
       ├── bbb.png
       └── ...
   └── testB
       ├── ccc.jpg 
       ├── ddd.png
       └── ...
```

## Training
```bash
python train.py --name EXP_NAME --d_path YOUR_DATASET_NAME
```

## Inference
Our notebook provides a comprehensive demo of both image and video translation. Pretrained model is automatically loaded.


## Citation
If you use this code or ideas from our paper, please cite our paper:
```
```

## Acknowledgments
This code borrows heavily from [StyleGan2 by rosalinity](https://github.com/rosinality/stylegan2-pytorch) and partly from [UGATIT](https://github.com/znxlwm/UGATIT-pytorch).
