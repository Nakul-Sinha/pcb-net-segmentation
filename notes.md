# Notes - Circuit Board Net Trace Segmentation

Working log of challenge facts, design decisions, validation, and submissions.

## Challenge facts
- Task: seed conditioned binary instance segmentation. Given a top view RGB crop, an xray like view, and a one dot seed marker, segment the full electrical net connected to the seed (traces, pads, vias, branches) and exclude all other nets.
- Inputs per sample: top.png (192x192 RGB), xray.png (192x192 near grayscale), seed_mask.png (192x192 binary, single 97 px disk).
- Output: submission.csv with columns image_id, mask_rle. RLE is 1 indexed, row major, start length pairs. Empty string means empty mask. Decoded shape must be 192x192.
- Metric: 0.45 dice + 0.25 thin trace recall + 0.20 endpoint connectivity + 0.10 boundary F1. Maximize, range [0,1].
- Data: 260 train, 160 test. Fully synthetic. Public data only, no external PCB data, no external API at inference, do not modify grade.py.

## Data facts measured directly from public_circuit
- Target net is always a single 8 connected component (100 percent of train). Seed centroid is always inside the target net (100 percent).
- Target net area is about 18 to 20 percent of all top copper, so the model must select 1 of about 5 nets. Seed conditioning is essential.
- About 20 percent of target net pixels are not copper colored on the top layer but are dark on the xray, so xray fusion is mandatory.
- Foreground is about 3.5 percent of pixels (range 0.7 to 9.3 percent), so region and topology losses beat plain BCE.
- Thinnest trace parts are about 4.3 px wide (median width about 5.7 px), so training stays at native 192 with no downsampling.
- top copper and xray darkness correlate at about 0.862, so the two views are pixel co registered and early channel fusion is clean.

## Approach (see Approach.md for full detail and citations)
- Input: 6 channel early fusion tensor [top R, top G, top B, xray gray, seed disk, seed gaussian].
- Model: timm pretrained encoder (default resnet34) plus a custom U-Net decoder, output a single logit map at native 192.
- Loss: composite BCE + soft Dice + Tversky (false negative weighted, alpha 0.3 beta 0.7) + soft clDice (ramped over the first half of training). Maps directly to the metric: Dice for area, Tversky and clDice for thin recall and connectivity.
- Training: 5 fold cross validation, AdamW, cosine schedule with warmup, EMA, hand rolled augmentation (D4, affine, light elastic, photometric on image channels only, identical geometry applied to all channels and mask).
- Inference: per fold probability bagging plus D4 flip TTA, averaged before thresholding.
- Post processing: hysteresis dual threshold on the probability map, force the seed region in, small morphological close to bridge sub pixel gaps, then keep only the connected component that contains the seed (8 connectivity). This directly serves the connectivity and thin recall terms and removes foreign nets.

## Compliance
- Learned deep model only. No hardcoding, no metadata or id shortcuts, no leaderboard probing.
- solution.py reads only DATA_ROOT (default ./dataset/public) and writes OUT_DIR/submission.csv (default ./working). No comments in the official code per project instruction; reasoning lives in Approach.md and notes.md.
- ImageNet pretrained encoder via timm, same family of pretrained download the previous Eris solution used. Final recipe is intended to reproduce on A10G class GPU.

## Validation
- RLE codec round trips all 40 sampled train masks. Convention verified: pixel (0,0) -> "1 1", pixel (0,1) -> "2 1", pixel (1,0) -> "193 1". Empty mask -> empty string.
- CPU smoke test (FAST=1, PRETRAINED=0, resnet18, 1 fold, 1 epoch, 24 train, 8 test) runs end to end: train, OOF metric, TTA inference, post process, RLE, write submission. Output passed the strict schema validator on the matching id subset.
- Smoke OOF score about 0.15 with a 1 epoch untrained model, as expected. Real metric comes from the full GPU run.

## Kaggle run infrastructure (kaggle_run driver under eris/kaggle_run)
- Data is uploaded once as a private dataset (single zip, notebook auto extracts if needed). ResNet34 weights are uploaded as a second private dataset so the kernel needs no internet for weights.
- solution.ipynb is generated from solution_core via make_notebook.py and pushed as a GPU kernel that reads /kaggle/input and writes /kaggle/working/submission.csv.
- Environment fixes found while bringing up the kernel:
  1. Kaggle GPU and internet require phone verification, otherwise the kernel silently runs on CPU with no network.
  2. Kaggle pre-installed torch 2.10 dropped Pascal sm_60, so a P100 raises "no kernel image is available". Fix: pin machine_shape NvidiaTeslaT4 (sm_75 is supported) and select fp16 AMP when bf16 is unsupported.

## Run log
- Run 1 (baseline): resnet34, 5 fold, 80 epochs, img 192, batch 16, composite loss, D4 TTA, hysteresis 0.35/0.55 + keep seed component. T4, about 9.5 min per fold.
  - OOF score 0.1738 (dice 0.170, thin recall 0.237, connectivity 0.119, boundary F1 0.141). Submission valid, 160 rows, all non empty, 109413 fg px.
  - Diagnosis: underfitting. Loss only fell 1.70 to 1.01 over 80 epochs because there are only 13 steps per epoch (208 train / batch 16). Per fold variance is high (fold 2 over predicts, others under predict), which is the threshold sensitivity expected from an undertrained, poorly calibrated model. This is a training budget problem, not a pipeline bug.

## Plan to the best submission (highest ROI first)
1. Fix underfitting: many more gradient steps (raise epochs and effective steps), higher peak LR, RAM cache decoded images and more dataloader workers to stop the GPU starving, lighter elastic and photometric augmentation so 208 images can be fit.
2. Tune thresholds (HI_THR, LO_THR), close kernel, and Tversky alpha beta on OOF once the model fits.
3. Stronger or second backbone (efficientnet_b3, mit_b2 / SegFormer) for a probability averaged ensemble.
4. Optional cbDice or boundary loss for the 0.10 boundary term if boundary F1 lags.

## Submissions
- Run 1 baseline committed. OOF score 0.1738. Next: address underfitting per the plan above.
