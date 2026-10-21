# Dual-Stream Deepfake Forensics and Real-Time Human Activity Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23523972.svg)](https://doi.org/10.5281/zenodo.23523972)

Reproducible implementation of a dual-stream video-surveillance system in which one lightweight convolutional neural network (six convolutional blocks, 183,682 parameters, about 1.03 GFLOPs per 256×256 frame) is trained separately for two tasks that run on the same video frames:

* **deepfake visual forensics**: real vs. GAN-generated face images;
* **human activity recognition (HAR)**: seven activities in surveillance-style video.

The repository contains the notebook that produces every result, the trained model weights, the training logs and the recorded outputs.

## Contents

| Path | Content |
|---|---|
| `notebooks/DualStream_Deepfake_HAR_Experiments.ipynb` | Complete experimental pipeline (Google Colab, GPU) |
| `models/` | Trained weights (Keras `.weights.h5`): `deepfake_final` (standard augmentation), `deepfake_robust_final` (robust augmentation), `har_final` (7 classes) |
| `training_logs/` | Per-epoch training/validation accuracy and loss; selected epoch and training settings |
| `results/results.json` | All recorded results (metrics, confidence intervals, fusion runs, cross-dataset tests, pipeline timings, hardware) |
| `results/B2_calibration.json` | Cross-dataset threshold calibration with a small labelled target sample |
| `results/results_tables.tex` | Result tables as LaTeX rows |
| `results/*_test_scores.npy` | Labels and scores of the 20,000 deepfake test images (both models) |
| `results/figures/`, `results/paper_figures/` | Figures as produced by the notebook and as used in the article |

## Datasets

All data are public and must be downloaded from their sources:

* **140k Real and Fake Faces** (Kaggle, xhlulu): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces. Published split used unchanged: 100,000 training, 20,000 validation and 20,000 test images, balanced.
* **Hard Fake vs Real Faces** (Kaggle, hamzaboulahia): https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces. 1,289 images (700 fake, 589 real), used only for cross-dataset testing.
* **Human Activity Recognition – Video Dataset** (Kaggle, Rajput et al., doi:10.34740/KAGGLE/DSV/5722068): https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset. 1,113 videos in seven classes, split by video (779 / 167 / 167, stratified, seed 42).

## Main results

| Experiment | Result |
|---|---|
| Deepfake stream, 20,000 test images (standard recipe) | accuracy 96.62% [95% CI 96.36–96.86], sensitivity 97.18%, specificity 96.06%, F1 96.64%, AUC 0.995 |
| Deepfake stream, robust recipe | accuracy 96.31% [96.05–96.56], AUC 0.994 |
| Cross-dataset, Hard Fake vs Real Faces, zero-shot | AUC 0.935; accuracy 66.0% at threshold 0.5 (sensitivity 37.7%, specificity 99.7%) |
| Cross-dataset, threshold calibrated on 20 / 644 labelled target images | accuracy 84.6 ± 2.1% / 86.2 ± 1.0% |
| Deep-feature baselines (linear SVM, 12 runs) | best single feature set ResNet50 fc1000: 85.75 ± 0.22%; no fused configuration exceeded it (Wilcoxon p < 0.001) |
| HAR stream, 167 held-out videos | 98.0% frame-level, 98.8% video-level accuracy |
| Full dual-stream pipeline, 30-FPS video, both streams on every frame | 36.6 FPS (NVIDIA A100), 32.7 FPS (12-thread Xeon CPU) |
| Same, face detection every 5th frame, HAR every 2nd frame | 141.3 FPS (A100), 97.5 FPS (12 threads), 43.8 FPS (1 thread) |

Known limitation: the deepfake detector does not generalise zero-shot to fake faces from an unseen source; its ranking ability largely transfers (AUC 0.935) but its decision threshold does not, and a small labelled sample of the new source is needed to recalibrate it.

## How to run

1. Upload the three dataset archives to Google Drive under `MyDrive/Datasets/HAR/` as `Real_Fake_Faces.zip`, `Hard_Fake_Real_Faces.zip` and `HAR_datasets.zip` (or edit the paths in the first code cell).
2. Open the notebook in Google Colab with a GPU runtime (an A100 was used for the reported results) and run all cells. `TIER = "quick"` runs every part on subsets for an end-to-end check; `TIER = "full"` reproduces the reported results.
3. Results, tables and figures are written to `MyDrive/Outputs/DualStream/results`; model checkpoints to `MyDrive/Outputs/DualStream/checkpoints`. Large temporary data stay on the Colab disk. An interrupted run resumes from its last checkpoint.

Every iterative procedure stops by an explicit rule: training stops when validation accuracy gains less than 0.5 points over 8 epochs; bootstrap resampling, repeated fusion runs, calibration splits and timing loops stop when their estimates have converged. The robust deepfake model and the HAR model were trained under the training rule; the standard deepfake model was stopped manually after 54 epochs (selected epoch 52, chosen on validation data), when validation accuracy had improved by less than one point over the preceding ten epochs.

To use the trained weights without retraining, build the network with `build_cnn(2)` (deepfake) or `build_cnn(7)` (HAR) as defined in the notebook and call `load_weights("models/<name>.weights.h5")`. Inputs are RGB images resized to 256×256 and scaled to [0, 1]; for the deepfake model, output 0 is the probability of *real* and output 1 the probability of *fake*; for the HAR model, outputs follow the alphabetical order of the seven class names.

## Software

Python 3.13, TensorFlow 2.21, OpenCV 4.14 (YuNet face detector from the OpenCV model zoo), PyTorch/torchvision (ImageNet feature extractors), RAPIDS cuML (linear SVM), scikit-learn, SciPy.

## License

Code: MIT License (see `LICENSE`). The datasets are not redistributed here and remain under the terms of their providers.

## Citation

Archived release v1.0.0: https://doi.org/10.5281/zenodo.23523972 (see also `CITATION.cff`).
