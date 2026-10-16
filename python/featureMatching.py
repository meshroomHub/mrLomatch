from pyalicevision import matchingImageCollection as avmic
from pyalicevision import matching as avmatch
from pyalicevision import system as avsys

from loma.loma import filter_matches

from common import *

import os
import logging


def load_view_features(featuresFolder, viewId):
    """
    Load the LoMa keypoints/descriptors written by LomaFeatureExtraction.

    Returns:
        (keypoints tensor (N, 2), descriptions tensor (N, D), model name)
    """
    data = np.load(os.path.join(featuresFolder, f"{viewId}.loma.npz"))
    keypoints = torch.from_numpy(data["keypoints"])
    descriptions = torch.from_numpy(data["descriptions"].astype(np.float32))
    return keypoints, descriptions, str(data["model"])

def compute_matches(inputSfMData, featuresFolder, imagePairsList, filterThreshold, matchesFolder, rangeIteration, rangeBlocksCount):
    """
    Match LoMa features on a list of image pairs and save AliceVision matches.

    Args:
        inputSfMData (str): Path to the input SfM data file.
        featuresFolder (str): Output folder of LomaFeatureExtraction.
        imagePairsList (str): Path to the file listing image pairs to process.
        filterThreshold (float): Minimal match score.
        matchesFolder (str): Output directory for match files.
        rangeIteration (int): Index of the current processing block.
        rangeBlocksCount (int): Total number of processing blocks.
    """
    iinfos = get_imageinfos_from_sfmdata(inputSfMData)

    # Load the image pairs to process (referenceView, matchingView)
    plist = avmic.PairSet()
    if not avmic.loadPairsFromFile(imagePairsList, plist, False):
        raise RuntimeError("Error in image pairs list loading")
    logging.info(f"Loaded {len(plist)} pairs from list")

    # Filter out the pairs with views not in current sfmData
    pairsToProcess = [pair for pair in plist if pair[0] in iinfos and pair[1] in iinfos]
    logging.info(f"Kept {len(pairsToProcess)} filtered pairs from list")

    (valid, rangeStart, rangeEnd) = avsys.rangeComputation(rangeIteration, rangeBlocksCount, len(pairsToProcess))
    if not valid:
        logging.error("Range is out of bounds.")
        return

    pairsToProcess = pairsToProcess[rangeStart:rangeEnd]
    logging.info(f"Processing elements {rangeStart} to {rangeEnd}")

    # Load features of all the views used in this range
    viewIds = set([pair[0] for pair in pairsToProcess] + [pair[1] for pair in pairsToProcess])
    features = dict()
    modelNames = set()
    for viewId in viewIds:
        keypoints, descriptions, modelName = load_view_features(featuresFolder, viewId)
        features[viewId] = (keypoints, descriptions)
        modelNames.add(modelName)

    global_matches = avmatch.PairwiseMatches()

    if len(pairsToProcess) > 0:
        if len(modelNames) != 1:
            raise RuntimeError(f"Features were extracted with different models: {modelNames}")
        modelName = modelNames.pop()

        logging.info(f"Loading model {modelName}")
        model = load_model(modelName)

        for (referenceId, otherId) in pairsToProcess:
            kA, dA = features[referenceId]
            kB, dB = features[otherId]

            with torch.inference_mode():
                scores = model(kA[None], kB[None], dA[None], dB[None])["scores"]
                m0, *_ = filter_matches(scores, filterThreshold)

            m0 = m0[0].cpu().numpy()
            validIds = np.where(m0 > -1)[0]

            matches = avmatch.IndMatches()
            for idA in validIds:
                matches.append(avmatch.IndMatch(int(idA), int(m0[idA])))

            logging.info(f"Pair {referenceId}-{otherId}: {len(validIds)} matches")

            perdesc = avmatch.MatchesPerDescType()
            perdesc[avmatch.EImageDescriberType_ROMA] = matches
            global_matches[avmatch.Pair(referenceId, otherId)] = perdesc

    avmatch.Save(global_matches, matchesFolder, "txt", False, f"{rangeIteration}_")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(prog='lomaFeatureMatching')

    parser.add_argument('--inputSfMData', type=str, help='')
    parser.add_argument('--featuresFolder', type=str, help='')
    parser.add_argument('--imagePairsList', type=str, help='')
    parser.add_argument('--filterThreshold', type=float, help='', default=0.1)
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--rangeIteration', type=int, help='', default=0)
    parser.add_argument('--rangeBlocksCount', type=int, help='', default=1)
    parser.set_defaults(func=compute_matches)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(inputSfMData=args.inputSfMData,
                  featuresFolder=args.featuresFolder,
                  imagePairsList=args.imagePairsList,
                  filterThreshold=args.filterThreshold,
                  matchesFolder=args.output,
                  rangeIteration=args.rangeIteration,
                  rangeBlocksCount=args.rangeBlocksCount)
    else:
        parser.print_help()
