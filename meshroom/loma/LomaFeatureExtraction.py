__version__ = "1.2"

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
        desc.ListAttribute(
            name="masksFolders",
            label="Masks Folders",
            description="List of folders where mask files can be looked up (same filename as the image).\n"
                        "Features located where the mask is 0 are removed.",
            elementDesc=desc.File(
                name="masksFolder",
                label="Masks Folder",
                description="",
                value="",
            ),
            exposed=True,
        ),
        desc.ChoiceParam(
            name="masksExtension",
            label="Mask File Extension",
            description="Mask file extension.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
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
        desc.IntParam(
            name="detectorResolution",
            label="Detector Resolution",
            description="Size of the longest side of the image used by the keypoint detector (rounded down to a multiple of 8).\n"
                        "Higher values give more accurate keypoints at a higher GPU memory cost. Descriptors are always computed at 784x784.",
            value=1024,
            range=(512, 8192, 256),
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
