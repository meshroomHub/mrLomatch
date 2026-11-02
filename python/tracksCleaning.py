import glob
import os
import logging

import numpy as np


def load_matches(matchesFolders):
    """
    Load all the AliceVision '*.matches.txt' files of the given folders.

    Returns:
        a list of (viewA, viewB, describerType, (N, 2) int64 array of feature indices)
    """
    matches = []
    for folder in matchesFolders:
        for filename in sorted(glob.glob(os.path.join(folder, "*.matches.txt"))):
            with open(filename) as f:
                tokens = f.read().split()

            i = 0
            while i < len(tokens):
                viewA, viewB, descCount = int(tokens[i]), int(tokens[i + 1]), int(tokens[i + 2])
                i += 3
                for _ in range(descCount):
                    descType, count = tokens[i], int(tokens[i + 1])
                    i += 2
                    pairMatches = np.array(tokens[i:i + 2 * count], dtype=np.int64).reshape(count, 2)
                    i += 2 * count
                    matches.append((viewA, viewB, descType, pairMatches))

    return matches

def save_matches(matches, filename):
    """
    Save matches in the AliceVision '*.matches.txt' format.

    Args:
        matches: a list of (viewA, viewB, describerType, (N, 2) array of feature indices)
        filename: output file path.
    """
    perPair = dict()
    for (viewA, viewB, descType, pairMatches) in matches:
        if len(pairMatches) > 0:
            perPair.setdefault((viewA, viewB), []).append((descType, pairMatches))

    with open(filename, "w") as f:
        for (viewA, viewB), perDesc in perPair.items():
            f.write(f"{viewA} {viewB}\n{len(perDesc)}\n")
            for (descType, pairMatches) in perDesc:
                f.write(f"{descType} {len(pairMatches)}\n")
                np.savetxt(f, pairMatches, fmt="%d")

def clean_tracks(matches):
    """
    Keep the largest subset of matches which builds tracks without forks
    (a track never contains two features of the same view).

    Matches are merged greedily with a union-find, pairs with the most matches first.
    A match joining two tracks is rejected if both tracks already observe a common view.

    Args:
        matches: a list of (viewA, viewB, describerType, (N, 2) array of feature indices)

    Returns:
        the kept matches, with the same structure.
    """
    # Unique id for each (view, describer type, feature)
    views = sorted(set([m[0] for m in matches] + [m[1] for m in matches]))
    viewIndex = {viewId: i for i, viewId in enumerate(views)}
    descTypes = sorted(set([m[2] for m in matches]))
    descIndex = {descType: i for i, descType in enumerate(descTypes)}
    maxFeature = max([int(m[3].max()) for m in matches if len(m[3]) > 0], default=0) + 1

    keysA = []
    keysB = []
    for (viewA, viewB, descType, pairMatches) in matches:
        offset = descIndex[descType] * maxFeature
        stride = len(descTypes) * maxFeature
        keysA.append(viewIndex[viewA] * stride + offset + pairMatches[:, 0])
        keysB.append(viewIndex[viewB] * stride + offset + pairMatches[:, 1])

    allKeys, inverse = np.unique(np.concatenate(keysA + keysB), return_inverse=True)
    nodeCount = len(allKeys)
    nodeViews = (allKeys // (len(descTypes) * maxFeature)).tolist()

    parent = list(range(nodeCount))
    size = [1] * nodeCount
    # Bitset of the views observed by each track (only valid for roots)
    viewsMask = [1 << v for v in nodeViews]

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # Split the node ids back per pair
    totalA = sum(len(k) for k in keysA)
    nodesA = np.split(inverse[:totalA], np.cumsum([len(k) for k in keysA])[:-1])
    nodesB = np.split(inverse[totalA:], np.cumsum([len(k) for k in keysB])[:-1])

    order = sorted(range(len(matches)), key=lambda i: len(matches[i][3]), reverse=True)

    keptMatches = [None] * len(matches)
    for i in order:
        keep = np.zeros(len(nodesA[i]), dtype=bool)
        for j, (a, b) in enumerate(zip(nodesA[i].tolist(), nodesB[i].tolist())):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                keep[j] = True
                continue
            if viewsMask[ra] & viewsMask[rb]:
                continue
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            viewsMask[ra] |= viewsMask[rb]
            viewsMask[rb] = 0
            keep[j] = True

        (viewA, viewB, descType, pairMatches) = matches[i]
        keptMatches[i] = (viewA, viewB, descType, pairMatches[keep])

    return keptMatches

def compute_tracks_cleaning(matchesFolders, outputFolder):
    """
    Remove the matches which would create tracks with forks.

    Args:
        matchesFolders (list[str]): Folders containing the input matches.
        outputFolder (str): Output directory for the cleaned matches.
    """
    matches = load_matches(matchesFolders)
    inputCount = sum(len(m[3]) for m in matches)
    logging.info(f"Loaded {inputCount} matches on {len(matches)} pairs")

    if inputCount == 0:
        keptMatches = matches
    else:
        keptMatches = clean_tracks(matches)

    keptCount = sum(len(m[3]) for m in keptMatches)
    logging.info(f"Kept {keptCount} matches ({inputCount - keptCount} removed)")

    save_matches(keptMatches, os.path.join(outputFolder, "0.matches.txt"))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='lomaTracksCleaning')

    parser.add_argument('--matchesFolders', nargs='*', default=[], help='')
    parser.add_argument('--output', type=str, help='')
    parser.add_argument('--verboseLevel', type=str, help='', default="info",
                        choices=["fatal", "error", "warning", "info", "debug", "trace"])
    parser.set_defaults(func=compute_tracks_cleaning)

    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s',
                        level="DEBUG" if args.verboseLevel == "trace" else args.verboseLevel.upper())

    if hasattr(args, 'func'):
        args.func(matchesFolders=args.matchesFolders,
                  outputFolder=args.output)
    else:
        parser.print_help()
