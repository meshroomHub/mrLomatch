__version__ = "1.0"

from pathlib import Path
from meshroom.core import desc

class LomaTracksCleaning(desc.CommandLineNode):

    category = "LOMA"
    documentation = """
    Remove the matches which would create tracks with forks (several features of the same view in one track).
    Matches are merged greedily, pairs with the most matches first, and a match is rejected if it joins two tracks
    observing a common view. Use it between GeometricFilter and TracksBuilding.
    """
    cpu = desc.Level.NORMAL
    ram = desc.Level.INTENSIVE

    exePath = (Path(__file__).absolute().parent.parent.parent / "python" / "tracksCleaning.py").as_posix()

    commandLine = "python " + exePath + " {allParams}"

    inputs = [
        desc.ListAttribute(
            name="matchesFolders",
            label="Matches Folders",
            description="Folders containing the matches to clean.",
            elementDesc=desc.File(
                name="matchesFolder",
                label="Matches Folder",
                description="",
                value="",
            ),
            exposed=True,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Matches Folder",
            description="Output folder for cleaned matches.",
            value="{nodeCacheFolder}",
        ),
    ]
