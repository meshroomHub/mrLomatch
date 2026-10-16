__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class LomaFeatureMatching(desc.CommandLineNode):

    category = "LOMA"
    documentation = """
    Match the features computed by LomaFeatureExtraction on a list of image pairs using the LoMa matcher.
    The model used is the one used for extraction.
    """
    size = avpar.DynamicViewsSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=200)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "featureMatching.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.File(
            name="featuresFolder",
            label="Features Folder",
            description="Output folder of LomaFeatureExtraction.",
            value="",
        ),
        desc.File(
            name="imagePairsList",
            label="Image Pairs",
            description="Path to a file which contains the list of image pairs to match.",
            value="",
        ),
        desc.FloatParam(
            name="filterThreshold",
            label="Filter Threshold",
            description="Minimal match score to keep a match.",
            value=0.1,
            range=(0.0, 1.0, 0.01),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Matches Folder",
            description="Output folder for matches.",
            value="{nodeCacheFolder}",
        ),
    ]
