# LoMa Nodes

Meshroom nodes using [LoMa](https://github.com/davnords/LoMa) (rez package `lomatch`) as a replacement for FeatureExtraction + FeatureMatching.

## Meshroom Nodes

### LomaFeatureExtraction
Detect and describe keypoints for each view of the input sfmData. Writes `<viewId>.roma.feat/.desc` (AliceVision features, `roma` describer type) and `<viewId>.loma.npz` (LoMa keypoints/descriptors used for matching).

### LomaFeatureMatching
Match the extracted features on a list of image pairs with the LoMa matcher. Writes AliceVision matches (`roma` describer type).

## Typical graph

CameraInit → LomaFeatureExtraction → ImageMatching → LomaFeatureMatching → StructureFromMotion

- ImageMatching: `featuresFolders` = LomaFeatureExtraction output, method `Exhaustive` or `Sequential` (vocabulary tree requires SIFT).
- StructureFromMotion: `featuresFolders` = LomaFeatureExtraction output, `matchesFolders` = LomaFeatureMatching output, `describerTypes` = `roma`.

## Weights
LoMa, DaD, DeDoDe and DINOv2 weights are downloaded by torch.hub on first run (`~/.cache/torch/hub`, or `$TORCH_HOME/hub`).
