__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc
from pyalicevision import parallelization as avpar

class LomaFeatureExtraction(desc.CommandLineNode):

    category = "LOMA"
    documentation = """
    Detect and describe keypoints with LoMa (DaD detector + DeDoDe descriptor) for each view of the sfmData.
    Features are saved with the 'roma' describer type, LoMa descriptors are saved alongside for LomaFeatureMatching.
    """
    size = avpar.DynamicViewsSize('inputSfMData')
    gpu = desc.Level.INTENSIVE

    parallelization = desc.Parallelization(blockSize=50)
    commandLineRange = "--rangeIteration {rangeIteration} --rangeBlocksCount {rangeBlocksCount}"

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "featureExtraction.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.File(
            name="inputSfMData",
            label="SfMData",
            description="Input SfMData file.",
            value="",
        ),
        desc.ChoiceParam(
            name="model",
            label="Model",
            description="LoMa model. loma_B is a good default, loma_G is the most accurate, loma_R is rotation invariant.",
            value="loma_B",
            values=["loma_B128", "loma_B", "loma_L", "loma_G", "loma_R"],
            exclusive=True,
        ),
        desc.IntParam(
            name="numKeypoints",
            label="Number of Keypoints",
            description="Number of keypoints detected per image.",
            value=2048,
            range=(256, 16384, 256),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Features Folder",
            description="Output folder for features and LoMa descriptors.",
            value="{nodeCacheFolder}",
        ),
    ]
