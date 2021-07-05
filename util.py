import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, autograd
import os
import matplotlib.pyplot as plt


google_drive_paths = {
    "GNR_checkpoint.pt": "https://drive.google.com/uc?id=1IMIVke4WDaGayUa7vk_xVw1uqIHikGtC",
    "GNR_checkpoint_new.pt": "https://drive.google.com/uc?id=1PQ_SRLfFsXO_9z_OW5H9gKhhmIMn7H-p",
}

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

def shuffle_batch(x):
    return x[torch.randperm(x.size(0))]

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    loss = 0
    for real, fake in zip(real_pred, fake_pred):
        real_loss = F.softplus(-real)
        fake_loss = F.softplus(fake)
        loss += real_loss.mean() + fake_loss.mean() 

    return loss


def d_r1_loss(real_pred, real_img):
    grad_penalty = 0
    for real in real_pred:
        grad_real, = autograd.grad(
            outputs=real.mean(), inputs=real_img, create_graph=True, only_inputs=True
        )
        grad_penalty += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred, weights):
    loss = 0
    for fake, weight in zip(fake_pred, weights):
        loss += weight*F.softplus(-fake).mean()

    return loss / len(fake_pred)

def display_image(image, size=None, mode='nearest', unnorm=False, title=''):
    # image is [3,h,w] or [1,3,h,w] tensor [0,1]
    if image.is_cuda:
        image = image.cpu()
    if size is not None and image.size(-1) != size:
        image = F.interpolate(image, size=(size,size), mode=mode)
    if image.dim() == 4:
        image = image[0]
    image = image.permute(1, 2, 0).detach().numpy()
    plt.figure()
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)

def normalize(x):
    return ((x+1)/2).clamp(0,1)

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.
    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def truncate(x, truncation, mean_style):
    return truncation*x + (1-truncation)*mean_style
