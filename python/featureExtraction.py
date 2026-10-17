from pyalicevision import feature as avfeat
from pyalicevision import system as avsys
from pyalicevision import image as avimage

from loma.loma import to_pixel_coords

from common import *

import os
import logging


def save_features(outputFolder, viewId, pixelCoords):
    """
    Save the keypoints as AliceVision 'roma' features (positions only, dummy descriptors).

    Args:
        outputFolder (str): output directory.
        viewId (int): view identifier.
        pixelCoords (numpy.ndarray): (N, 2) keypoints in original image pixel coordinates.
    """
    regions = avfeat.RomaRegions()
    for coord in pixelCoords:
        regions.Descriptors().append(avfeat.RomaDescriptor())
        regions.Features().append(avfeat.PointFeature(float(coord[0]), float(coord[1]), 1.0, 0.0))

    regions.Save(f"{outputFolder}/{viewId}.roma.feat", f"{outputFolder}/{viewId}.roma.desc")

def parse_masks_folders(masksFolders):
    """
    Normalise the masks-folder argument into a deduplicated list of non-empty path strings.
    """
    if masksFolders is None:
        return []
    if isinstance(masksFolders, str):
        masksFolders = [masksFolders]

    folders = []
    for folder in masksFolders:
        folder = folder.strip()
        if folder and folder not in folders:
            folders.append(folder)
    return folders

def load_mask(masksFolders, masksExtension, imagePath, width, height):
    """
    Load the mask of an image, looking for '<imageName>.<ext>' in each masks folder
    (same convention as MatchMasking). If several masks are found, they are combined (minimum).

    Returns:
    a (height, width) numpy array, or None if no mask is found.
    """
    stem = os.path.splitext(os.path.basename(imagePath))[0]
    mask = None

    for folder in masksFolders:
        maskPath = os.path.join(folder, f"{stem}.{masksExtension}")
        if not os.path.exists(maskPath):
            continue

        logging.info(f"Found mask at {maskPath}")
        maskLarge = avimage.Image_uchar()
        avimage.readImage(maskPath, maskLarge, avimage.ImageReadOptions(avimage.EImageColorSpace_NO_CONVERSION))
        maskResized = avimage.Image_uchar()
        avimage.resampleImage(width, height, maskLarge, maskResized, False)
        current = np.squeeze(maskResized.getNumpyArray())
        mask = current if mask is None else np.minimum(mask, current)

    return mask

def unmasked_indices(pixelCoords, mask):
    """
    Indices of the keypoints which are not masked (mask value is not 0).
    Keypoints outside of the mask are kept.
    """
    x = pixelCoords[:, 0].astype(np.int64)
    y = pixelCoords[:, 1].astype(np.int64)
    inside = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])

    masked = np.zeros(len(pixelCoords), dtype=bool)
    masked[inside] = mask[y[inside], x[inside]] == 0

    return np.nonzero(~masked)[0]

def compute_features(inputSfMData, modelName, numKeypoints, detectorResolution, masksFolders, masksExtension, outputFolder, rangeIteration, rangeBlocksCount):
    """
    Detect and describe LoMa keypoints for each view of the sfmData.

    For each view in the assigned range, the function writes:
    - ``<viewId>.roma.feat/.desc``: AliceVision features (used by SfM),
    - ``<viewId>.loma.npz``: normalized keypoints and LoMa descriptors
      (used by LomaFeatureMatching). Row i corresponds to feature i.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        modelName (str): LoMa preset name.
        numKeypoints (int): Number of keypoints to detect per image.
        detectorResolution (int): Longest side of the image given to the detector.
        masksFolders (list[str]): Folders containing masks used to filter features (optional).
        masksExtension (str): File extension of the masks.
        outputFolder (str): Output directory.
        rangeIteration (int): Index of the current processing block.
        rangeBlocksCount (int): Total number of processing blocks.
    """
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)
    viewIds = list(iinfos.keys())

    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(viewIds))
    if not valid:
        logging.error("Range is out of bounds.")
        return

    viewIds = viewIds[rangeStart:rangeEnd]
    logging.info(f"Processing views {rangeStart} to {rangeEnd}")

    masksFolders = parse_masks_folders(masksFolders)

    logging.info(f"Loading model {modelName}")
    model = load_model(modelName)

    for viewId in viewIds:
        info = iinfos[viewId]
        logging.info(f"Extracting features for view {viewId} ({info.path})")

        pil_im = load_image_to_pil(info.path)
        W, H = pil_im.size
        detectorImage, descriptorImage = prepare_images(pil_im, model, detectorResolution)

        with torch.inference_mode():
            keypoints = model._detector.detect(detectorImage, num_keypoints=numKeypoints)["keypoints"]
            descriptions = model._descriptor.describe_keypoints(descriptorImage, keypoints)["descriptions"]

        keypoints = keypoints[0]
        descriptions = descriptions[0]
        pixelCoords = to_pixel_coords(keypoints, H, W).cpu().numpy()

        mask = load_mask(masksFolders, masksExtension, info.path, W, H)
        if mask is not None:
            selected = unmasked_indices(pixelCoords, mask)
            logging.info(f"Mask removed {pixelCoords.shape[0] - len(selected)} features")
            pixelCoords = pixelCoords[selected]
            selected = torch.from_numpy(selected).to(keypoints.device)
            keypoints = keypoints[selected]
            descriptions = descriptions[selected]

        save_features(outputFolder, viewId, pixelCoords)

        np.savez(
            os.path.join(outputFolder, f"{viewId}.loma.npz"),
            keypoints=keypoints.float().cpu().numpy(),
            descriptions=descriptions.half().cpu().numpy(),
            model=modelName,
        )

        logging.info(f"{pixelCoords.shape[0]} features saved")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(prog='lomaFeatureExtraction')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--model', type=str, help='', default="loma_B")
    parser.add_argument('--numKeypoints', type=int, help='', default=2048)
    parser.add_argument('--detectorResolution', type=int, help='', default=1024)
    parser.add_argument('--masksFolders', nargs='*', default=None, help='')
    parser.add_argument('--masksExtension', type=str, help='', default="exr")
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_features)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                  modelName=args.model,
                  numKeypoints=args.numKeypoints,
                  detectorResolution=args.detectorResolution,
                  masksFolders=args.masksFolders,
                  masksExtension=args.masksExtension,
                  outputFolder=args.output,
                  rangeIteration=args.rangeIteration,
                  rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
