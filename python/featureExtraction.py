from pyalicevision import feature as avfeat
from pyalicevision import system as avsys

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

def compute_features(inputSfMData, modelName, numKeypoints, outputFolder, rangeIteration, rangeBlocksCount):
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

    logging.info(f"Loading model {modelName}")
    model = load_model(modelName)

    for viewId in viewIds:
        info = iinfos[viewId]
        logging.info(f"Extracting features for view {viewId} ({info.path})")

        pil_im = load_image_to_pil(info.path)
        W, H = pil_im.size
        detectorImage, descriptorImage = prepare_images(pil_im, model)

        with torch.inference_mode():
            keypoints = model._detector.detect(detectorImage, num_keypoints=numKeypoints)["keypoints"]
            descriptions = model._descriptor.describe_keypoints(descriptorImage, keypoints)["descriptions"]

        pixelCoords = to_pixel_coords(keypoints[0], H, W).cpu().numpy()
        save_features(outputFolder, viewId, pixelCoords)

        np.savez(
            os.path.join(outputFolder, f"{viewId}.loma.npz"),
            keypoints=keypoints[0].float().cpu().numpy(),
            descriptions=descriptions[0].half().cpu().numpy(),
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
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_features)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                  modelName=args.model,
                  numKeypoints=args.numKeypoints,
                  outputFolder=args.output,
                  rangeIteration=args.rangeIteration,
                  rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
