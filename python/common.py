from pyalicevision import image as avimage
from pyalicevision import sfmData as avsfmData
from pyalicevision import sfmDataIO as avsfmDataIO

import argparse
import numpy as np
import torch
from PIL import Image

from loma.device import device

# Same size as the one used by DeDoDeDescriptor.read_image
DESCRIPTOR_SIZE = 784

class ImageInfo:
    def __init__(self):
        self.path = ""
        self.uid = 0
        self.width = 0
        self.height = 0

def get_imageinfos_from_sfmdata(inputSfMData):
    """
    Parse a sfmData and fill a dict of ImageInfo struct

    Parameters:
    inputSfMData : the filesystem path to the sfmData

    Returns:
    a dict (indexed by view uid) of ImageInfo descriptions
    """

    data = avsfmData.SfMData()
    ret = avsfmDataIO.load(data, inputSfMData, avsfmDataIO.VIEWS)
    if not ret:
        raise RuntimeError("Error with sfm data")

    views = data.getViews()

    #sort by frameid
    views = dict(sorted(views.items(), key=lambda v: v[1].getFrameId()))

    infos = dict()
    for (key, view) in views.items():
        item = ImageInfo()
        item.uid = key
        item.path = view.getImage().getImagePath()
        item.width = int(view.getImage().getWidth())
        item.height = int(view.getImage().getHeight())
        infos[key] = item

    return infos

def load_model(modelName):
    """
    Create a LoMa model given its preset name (loma_B128, loma_B, loma_L, loma_G, loma_R).
    Weights are downloaded by torch.hub if not already in cache.
    """
    from loma import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR

    presets = {
        "loma_B128": LoMaB128,
        "loma_B": LoMaB,
        "loma_L": LoMaL,
        "loma_G": LoMaG,
        "loma_R": LoMaR,
    }

    if modelName not in presets:
        raise RuntimeError(f"Unknown LoMa model {modelName}")

    return LoMa(presets[modelName]())

def load_image_to_pil(path):
    """ Load an image using pyalicevision (supports all AliceVision formats)

    Parameters:
    path : the filesystem path to the image
    """

    image = avimage.Image_RGBColor()
    optRead = avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION)
    avimage.readImage(path, image, optRead)

    return Image.fromarray(image.getNumpyArray())

def pil_to_tensor(pil_im):
    array = np.array(pil_im) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).float().to(device)[None]

def prepare_images(pil_im, model):
    """
    Replicate the preprocessing done by LoMa when given an image path
    (the tensor API of LoMa does not resize the images).

    Returns:
    (detector image tensor, descriptor image tensor)
    """
    detector = model._detector
    W, H = pil_im.size

    if detector.keep_aspect_ratio:
        scale = detector.resize / max(W, H)
        dw = int((scale * W) // 8 * 8)
        dh = int((scale * H) // 8 * 8)
    else:
        dw, dh = detector.resize, detector.resize

    detectorImage = pil_to_tensor(pil_im.resize((dw, dh)))
    descriptorImage = pil_to_tensor(pil_im.resize((DESCRIPTOR_SIZE, DESCRIPTOR_SIZE)))

    return detectorImage, descriptorImage

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")
