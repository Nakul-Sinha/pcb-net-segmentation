# Requirements — Circuit Board Net Trace Segmentation

What you need (compute, software, data, and skills) to **build, train, and run** the solution
described in [Approach.md](Approach.md). The challenge is intentionally small‑scale (192×192 images,
260 train / 160 test), so requirements are **modest** — a single mid‑range GPU is enough.

> **Run on the web, not a local app.** The local Python here ships **CPU‑only PyTorch**
> (`torch.cuda.is_available() == False`), so training locally would be painfully slow.
> Use a **web/cloud GPU notebook** (Google Colab / Kaggle Kernels / a cloud VM). See §3.

---

## 1. Compute requirements

### 1.1 Minimum (will work)
| Resource | Minimum | Notes |
|---|---|---|
| **GPU** | 1× NVIDIA, **≥ 6 GB VRAM** (e.g., T4, RTX 2060) | 192×192 + small backbones (resnet34/effb2) fit easily; even SimpleClick's largest ViT‑H runs at ~3.2 GB, so our small models need far less. |
| **CPU** | 4 cores | Data loading, augmentation, RLE, post‑processing. |
| **RAM** | 8 GB | Whole dataset is small (PNGs at 192×192). |
| **Disk** | ~2 GB | Dataset + checkpoints + logs. |

### 1.2 Recommended (comfortable, faster iteration)
| Resource | Recommended | Notes |
|---|---|---|
| **GPU** | 1× **T4 / P100 / RTX 3060–4090 / A10 / L4**, **12–16 GB VRAM** | Headroom for U‑Net++, larger batches, mixed precision, dual‑encoder fusion, and the **clDice loss overhead** (≈ +88% training time, ≈ +52% VRAM vs Dice‑only — budget for it). |
| **CPU** | 8+ cores | Faster albumentations pipeline (elastic/affine are CPU‑bound). |
| **RAM** | 16 GB | Cache decoded images in memory. |
| **Disk** | 5–10 GB | 5 CV folds × multiple backbones × checkpoints + TTA artifacts. |

### 1.3 Training‑time budget (rough, on a single T4‑class GPU)
- One model, 192×192, resnet34 U‑Net, ~100–150 epochs on 260 images: **~30–90 min**.
- **5‑fold CV** (the recommended setup): **~3–6 hours** total.
- Add **clDice** (+~88% time) and a second backbone for ensembling → **plan for ~1 GPU‑day** end‑to‑end including ablations.
- Inference on 160 test images with TTA + 5‑fold ensemble + post‑processing: **a few minutes**.

> **No multi‑GPU, no large‑memory accelerator, and no long training runs are needed.** This is a "small data, careful method" problem, not a "big compute" problem.

---

## 2. Software / environment

### 2.1 Core stack
| Package | Purpose | Notes |
|---|---|---|
| **Python 3.10–3.12** | runtime | 3.12 is present locally; any 3.10+ is fine. |
| **PyTorch (CUDA build)** ≥ 2.1 | training/inference | **Install the CUDA wheel**, not the CPU wheel that's currently local. Match the CUDA version to the GPU host (e.g., cu121/cu124). |
| **torchvision** | transforms, pretrained weights | matches the torch version. |
| **segmentation‑models‑pytorch (SMP)** | U‑Net/U‑Net++/FPN + ImageNet‑pretrained encoders | the backbone of the model zoo. https://segmentation-models-pytorch.readthedocs.io |
| **albumentations** | augmentation (flips, affine, elastic/grid distortion, photometric) | apply identical geometric transforms to top/xray/seed/mask. |
| **MONAI** | loss functions (Dice, Tversky, generalized Dice) + medical‑seg utilities | https://monai.io |
| **clDice (`pip install cldice` or jocpae/clDice)** | soft‑clDice topology loss | https://github.com/jocpae/clDice |
| **OpenCV (`opencv-python`)** | connected components, morphology, distance transform, CLAHE | already present locally (4.10). |
| **scikit‑image** | skeletonize, morphology, hysteresis threshold | `skimage.filters.apply_hysteresis_threshold`, `skimage.morphology`. |
| **SimpleITK** *(optional)* | Fast‑Marching / geodesic region growing | only if you use the FMM post‑processing refinement. |
| **NumPy, pandas** | arrays, CSV / RLE I/O | numpy already present (2.1). |
| **scipy** | distance transforms, connected components, fast‑marching (`skfmm` alt) | already present. |
| **tqdm, matplotlib** | progress + visualization/debugging | inspect predictions vs masks. |

### 2.2 Optional / nice‑to‑have
| Package | Purpose |
|---|---|
| **timm** | extra encoders (EfficientNet, MiT/SegFormer, ConvNeXt) for SMP. |
| **ttach** | test‑time augmentation wrappers. |
| **Weights & Biases / TensorBoard** | experiment tracking across folds/backbones. |
| **scikit‑fmm (`skfmm`)** | lightweight fast‑marching without SimpleITK. |
| **pytest** | unit‑test the RLE round‑trip and the local metric re‑implementation. |

### 2.3 Example install (on a CUDA host / cloud GPU)
```bash
# 1) PyTorch with CUDA (pick the index-url matching the host's CUDA; cu124 shown)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Segmentation + losses + augmentation
pip install segmentation-models-pytorch albumentations monai timm

# 3) Topology loss + classic CV post-processing
pip install opencv-python scikit-image scipy simpleitk scikit-fmm
# clDice: pip install cldice   (or clone https://github.com/jocpae/clDice)

# 4) Data / utils / tracking
pip install numpy pandas tqdm matplotlib ttach wandb pytest
```

> **Verify the GPU is actually used** before training: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print `True` and the GPU name. (Locally it prints `False` — that's the CPU‑only wheel.)

---

## 3. Where to run it (web compute — no local app)

The user constraint is explicit: **use the web, not the app route**. Recommended options, in order:

1. **Google Colab** — free T4 (or paid A100/L4). Mount the dataset (upload `public_circuit/` or pull from your storage), `pip install` the stack above, train. Simplest path to a working GPU.
2. **Kaggle Kernels/Notebooks** — free T4×2 / P100, 30 GB+ working disk, persistent datasets. Good if the challenge dataset can be attached as a Kaggle dataset. Note: Kaggle blocks internet during scored runs — ensure all pip installs/pretrained weights are cached, and remember **no external API at inference** is a challenge rule anyway.
3. **A cloud GPU VM** (AWS g4dn/g5, GCP, Lambda, Paperspace, RunPod, Vast.ai) — pick a **single T4/L4/A10** instance; the job is small. Best for unattended multi‑hour CV + ensembling runs.

For any option: a **single mid‑range GPU instance** is sufficient; do **not** provision multi‑GPU or high‑memory accelerators.

---

## 4. Dataset requirements (already provided)
Use **only** the public challenge files (no external PCB data — a hard rule):
- `train.csv` (260 rows: `image_id, top_path, xray_path, seed_mask_path, mask_path`)
- `test.csv` (160 rows: `image_id, top_path, xray_path, seed_mask_path`)
- `sample_submission.csv` (160 rows: `image_id, mask_rle`) — the required output format.
- `train/images/` (top + xray PNGs), `train/seed_masks/`, `train/masks/`
- `test/images/` (top + xray PNGs), `test/seed_masks/`

All images are **192×192 PNG**; total footprint is small (~tens of MB).

---

## 5. Deliverable / submission requirements
- Produce `submission.csv` with **exactly** the columns `image_id,mask_rle`.
- **One row per `image_id` in `test.csv`**, no duplicates, no foreign ids.
- `mask_rle` = **1‑indexed, row‑major** `start length` pairs; **empty string = empty mask**; decoded mask must be **192×192**.
- **Unit‑test the RLE round‑trip** (`decode(encode(M)) == M`) and confirm orientation/indexing against `sample_submission.csv` before submitting.

---

## 6. Extra / human requirements
- **Skills:** PyTorch segmentation, loss‑function engineering, classical CV post‑processing (morphology, connected components, distance transforms), and disciplined cross‑validation.
- **A local re‑implementation of the exact scoring metric** (Dice + thin‑trace recall + endpoint connectivity + boundary F1) for offline model selection — this is essential since the public leaderboard gives limited feedback.
- **Reproducibility:** fix random seeds (Python/NumPy/PyTorch), log the environment, and save per‑fold checkpoints so the ensemble is reconstructable.
- **Time budget:** ~1 GPU‑day of compute plus a few days of human iteration (baseline → loss/post‑processing ablations → fusion/ensemble) to climb the metric.
- **Compliance:** respect the "Data Use And Solution Constraints" — public data only, no external PCB images/datasets, no answer‑key reconstruction, no id/hash/template‑lookup exploits, **no external API at inference**, and don't modify `grade.py`.

---

## 7. Summary
| Question | Answer |
|---|---|
| GPU needed? | **Yes, 1 GPU** (≥6 GB min, 12–16 GB recommended). Local env is CPU‑only → use web/cloud. |
| How much compute? | **~1 GPU‑day** for full 5‑fold CV + ensemble + ablations. Inference: minutes. |
| Where to run? | **Web GPU**: Google Colab / Kaggle / single cloud GPU VM. Not the (unconfigured) app route. |
| Core libraries? | PyTorch (CUDA) + segmentation‑models‑pytorch + MONAI + albumentations + clDice + OpenCV/scikit‑image/SimpleITK. |
| Extra data? | **None allowed** — public challenge files only. |
| Hardest non‑compute requirement? | A faithful **local metric re‑implementation** + a **bug‑free RLE codec**. |
