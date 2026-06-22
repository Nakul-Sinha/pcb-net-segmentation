# Approach — Circuit Board Net Trace Segmentation

A detailed, research-backed solution strategy for the seed‑conditioned PCB net‑trace
segmentation challenge. Every recommendation below is grounded in **(a)** direct
measurement of the public dataset and **(b)** verified findings from a multi‑source
literature review (papers + GitHub repos are cited inline; see [References](#references)).

---

## 1. Problem understanding

### 1.1 What the task actually is
Given, for each board crop:

- `top.png` — 192×192 **RGB** top view (copper = gold shapes on a teal solder‑mask background, plus pseudo‑silkscreen noise),
- `xray.png` — 192×192 grayscale **x‑ray‑like** view that reveals internal/bottom‑layer copper and via connectivity not visible on top,
- `seed_mask.png` — 192×192 binary mask with a **single small dot** marking where the requested electrical net begins,

we must output a 192×192 binary mask of **the complete electrical net connected to the seed**
(connected copper traces + pads + vias + branches) and **exclude every other net**, even when
foreign traces visually cross or run close. Output is submitted as **run‑length encoding (RLE)**
(1‑indexed, row‑major, `start length` pairs).

This is **not** plain semantic "find all copper" segmentation. It is **seed‑conditioned (point‑promptable)
binary instance segmentation**: the seed selects *one* instance (net) out of many that all look identical.

### 1.2 The scoring metric drives every design decision
```
score = 0.45 · net_mask_Dice
      + 0.25 · thin_trace_recall          (recall on private centerline pixels)
      + 0.20 · endpoint_connectivity      (seed-connected component must cover terminal pads;
                                           size factor min(1, 2·A_target / A_component) penalizes big blobs)
      + 0.10 · boundary_F1                 (2-px tolerance)
```
Reading the weights:
- **0.45 Dice** → get the bulk region right (area overlap).
- **0.25 thin‑trace recall** → **do not miss thin traces**; recall (not precision) is scored, so under‑segmenting thin routes is heavily punished. This is the single most "winnable" differentiator.
- **0.20 endpoint connectivity** → the prediction must form **one connected component from the seed that reaches the terminal pads**, and must **not** be an over‑large blob (the size factor caps credit at `min(1, 2·A_target/A_component)`).
- **0.10 boundary F1** → clean edges within 2 px.

Three of the four terms (0.55 total) reward **connectivity + thin‑structure completeness**, not raw area. This dictates topology‑aware training and connectivity‑aware post‑processing.

### 1.3 Data facts measured directly from `public_circuit/` (these anchor the whole strategy)
| Measured property | Value | Consequence for the approach |
|---|---|---|
| Train / test samples | **260 / 160** | Tiny data → pretrained backbones, heavy augmentation, small models, cross‑validation, ensembling. |
| Image size, channels | 192×192; `top`=RGB, `xray`=RGB (near‑grayscale: max channel diff ≈ 38–54), `seed`/`mask`=L | Train at **native 192×192**; keep x‑ray as its own channel(s). |
| Target net is a single 8‑connected component | **100% of training masks** | Post‑process by keeping **only the seed's connected component**. This is a free, large win on the 0.20 term. |
| Seed centroid lies inside the target net | **100%** | The seed is a reliable interior anchor → seed‑driven region growing / component selection is safe. |
| Target‑net area ÷ total top‑copper area | **mean ≈ 0.20, median ≈ 0.18** | The model must pick **~1 of ~5 nets** → seed conditioning is essential; a copper detector alone scores poorly. |
| Net pixels **not** copper‑colored on top but dark on x‑ray | **≈ 20%** | **~1/5 of the net is only recoverable from the x‑ray** → multi‑modal fusion is mandatory, not optional. |
| `corr(top‑copper, xray‑darkness)` | **0.862** | The two views are **pixel‑co‑registered** → **early channel‑concatenation fusion is clean** (no registration needed). |
| Foreground fraction of the 192×192 image | **mean ≈ 3.5%, range 0.7–9.3%** | Severe class imbalance → region/topology losses (Dice/Tversky/clDice), **not** plain BCE. |
| Thinnest trace width (5th‑pctile along centerline) | **≈ 4.3 px** (median width ≈ 5.7 px) | Downsampling below 192 risks erasing thin traces → **do not downsample**; keep (or upsample) resolution. |
| Seed dot size | constant **97 px** disk | Encode seed deterministically as a disk/Gaussian channel. |
| Mask values | strictly {0, 255} | Clean binary targets. |

> Bottom line: it is a **promptable, multi‑modal, thin‑structure, connectivity‑scored** segmentation problem on a **tiny co‑registered synthetic dataset**. The winning system is a small pretrained encoder–decoder that ingests `[top RGB ⊕ xray ⊕ seed]`, is trained with a **composite topology+imbalance loss**, and is finished with a **seed‑anchored connectivity post‑processor**.

---

## 2. Solution architecture at a glance

```
            ┌──────────── INPUT TENSOR (early fusion, co-registered) ────────────┐
 top.png ──▶│ R G B                                                              │
xray.png ──▶│       Xgray (+ optional CLAHE / dark-metal channel)                │ ─▶ 192×192 × (4–6) ch
seed.png ──▶│                       Seed disk + Gaussian heatmap                 │
            └────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
              ImageNet-pretrained encoder–decoder  (U-Net / U-Net++ / FPN,
              encoder = ResNet34 / EfficientNet-b2 / MiT-b2; small = anti-overfit)
                                        │  (soft probability map p ∈ [0,1]^{192×192})
                                        ▼
        Composite loss:  Dice/BCE  +  soft-clDice  +  Tversky(FN-weighted)  +  Boundary
                                        │
                                        ▼  (inference: TTA + k-fold ensemble averaging)
              ┌──────────── CONNECTIVITY-AWARE POST-PROCESSING ───────────┐
              │ 1. hysteresis dual-threshold on p (high-seed, low-grow)    │
              │ 2. keep ONLY the connected component touching the seed      │
              │ 3. gentle morphological close to bridge sub-px gaps         │
              │ 4. (optional) seed-driven Fast-Marching / geodesic grow     │
              │ 5. size-factor guard (don't over-grow → caps the 0.20 term) │
              └─────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                          RLE encode (1-indexed, row-major) → submission.csv
```

---

## 3. Detailed strategy

### 3.1 Problem formulation: seed‑conditioned (promptable) segmentation
The model is a function `f(top, xray, seed) → mask`. The standard, low‑risk, and verified way to make
an encoder–decoder "promptable" is to **feed the prompt as an extra input channel** rather than inventing
a bespoke prompt mechanism:

- **DEXTR** adds an extra channel containing a 2‑D Gaussian centered on each prompt point and learns to map it to the matching object [DEXTR]. 
- **RITM** is a **forward‑pass‑only** (no inference‑time optimization) click model that concatenates the prior/external mask as an extra input channel and can be *initialized from an external mask, then corrected* — almost exactly our "seed mask as guidance channel" setup [RITM].
- **SimpleClick** encodes clicks as positive/negative **disk** maps concatenated with the prior segmentation, fused via patch embedding [SimpleClick].
- **IU‑Net** shows that adding foreground/background click channels with a weighted loss beats a vanilla U‑Net [IU‑Net].

> **Decision:** encode the seed as **both** a binary disk (radius matching the 97‑px dot) **and** a smooth 2‑D Gaussian heatmap channel, and concatenate them to the image. Both disk and Gaussian encodings are validated; using both costs nothing and gives the network a sharp anchor + a soft distance cue.

### 3.2 Multi‑modal fusion (top RGB + x‑ray)
The two views are **co‑registered (corr ≈ 0.862)** and ~20% of the net lives only in the x‑ray.

- **Early (input‑level) fusion = channel concatenation** is the simplest, most widely‑used baseline in multi‑modal medical segmentation and is **competitive precisely when modalities are co‑registered** [Fusion‑Survey]. It is the correct **default** here.
- **Mid/feature‑level fusion** (dual‑encoder streams merged at the bottleneck, e.g., **FuseNet**, **RSFNet**) *can* beat early fusion — **but only when the fusion mechanism is effective**, and on tiny datasets the gain is real‑but‑modest and **not always statistically significant** (one 157‑case study: dual‑encoder Dice 0.823 vs concat 0.788, **p>0.05**) [FuseNet][RSFNet][SmallData‑Fusion].
- If you do add a second encoder, two evidence‑backed tricks keep it overfitting‑safe: **(a)** confine modality‑specific convolutions to **only the first encoder block** so parameter count barely rises [SmallData‑Fusion]; **(b)** use an **asymmetric** (smaller) encoder for the information‑poorer grayscale x‑ray, mirroring thermal‑vs‑RGB practice [RSFNet]. Pixel‑level and feature‑level fusion are additive, not redundant [PIF‑MSFF].

> **Decision:** **Start with early fusion** — input tensor `[R, G, B, Xgray, seed_disk, seed_gauss]` (4–6 channels). Treat a **first‑block‑split dual encoder** (RGB stream + small x‑ray stream) as a *Phase‑2 upgrade* to A/B test, not the baseline. Given only 260 samples, do **not** jump straight to a heavy cross‑attention fusion transformer — the data won't support it.

Optional engineered channels that are cheap and on‑policy (computed from public images only): a **CLAHE‑enhanced x‑ray** and a **"dark‑metal" channel** (`255 − xray`) to make internal copper explicit.

### 3.3 Backbone & decoder
- Use a **U‑shaped encoder–decoder with an ImageNet‑pretrained encoder** via **`segmentation-models-pytorch` (SMP)** — pretrained encoders are the standard overfitting cure on tiny data [D‑LinkNet][SMP]. (Note: the popular "D‑LinkNet won DeepGlobe 2018 1st place" claim **did not survive verification** — rely on the *architecture/pretraining* facts, not the ranking.)
- **Decoder/family:** **U‑Net** or **U‑Net++** (dense skips help thin structures) or **FPN**. U‑Net++ tends to help fine detail at a small extra cost.
- **Encoder size (anti‑overfit is the priority):** start **small** — `resnet34`, `efficientnet‑b2`, or `mit_b2` (SegFormer). Interactive‑seg evidence shows small backbones are the realistic choice at this data scale (SimpleClick ViT‑xTiny = 3.72 M params is viable) [SimpleClick]. Avoid ViT‑H‑class models — they overfit 260 images.
- **Resolution:** train and infer at **native 192×192** (or pad to 224 for stride‑friendly encoders). **Never downsample** — thin traces are only ~4.3 px wide.
- **First‑conv adaptation:** replace the 3‑channel stem with a 4–6‑channel conv; copy pretrained RGB weights into the RGB channels and initialize the extra channels (x‑ray, seed) from the mean of the pretrained weights (standard transfer‑learning channel‑inflation).

**On SAM / SAM2:** **do not** use zero‑shot SAM — it explicitly *fails on continuous branching/tubular structures* (zero‑shot retinal‑vessel Dice ≈ 0.227, the same thin‑curvilinear class as PCB traces) and cannot match purpose‑built models [SAM‑Med]. A **fine‑tuned/prompt‑tuned** SAM‑class model *is* viable even on tiny data (SAM vessel Dice 0.227→0.773 after fine‑tuning on ~20 pairs; PTSAM trains reliably from 16 images) [SAM‑Med][PTSAM] — keep it as an **optional ensemble member**, not the primary model. (The headline "PTSAM matches SOTA with ~2,048 params" framing was **refuted** — don't rely on it.)

### 3.4 Loss design — explicitly mapped to the metric
Pixel‑wise losses alone produce **high‑precision/low‑recall, broken‑connection** masks — exactly the failure modes the metric punishes [clDice][Boundary]. Use a **composite loss**, each term targeting a metric component:

| Loss term | Targets metric term | Why (verified) |
|---|---|---|
| **Soft Dice** (or Dice+BCE) | 0.45 Dice | Region overlap; the volumetric backbone of the loss. Recommended as the stable base to combine others with [clDice]. |
| **soft‑clDice** (centerline Dice) | 0.25 thin recall + 0.20 connectivity | Computed on the intersection of masks with their morphological **skeleta**; training with it yields "more accurate connectivity, higher graph similarity, better volumetric scores" on tubular data [clDice]. Standard mix: `L = (1−α)(1−softDice) + α(1−soft‑clDice)`. *(Note: soft‑clDice improves but does **not guarantee** topology — the "guarantees homotopy" claim was refuted.)* |
| **Tversky** (FN‑weighted, e.g., α≈0.3 FP / β≈0.7 FN) | 0.25 thin recall + 0.20 connectivity | Designed to fix imbalance and **trade precision for recall** by penalizing false negatives harder — directly serving the recall‑weighted score [Tversky]. Tune α/β empirically (gains over Dice can be marginal). |
| **Boundary loss** (distance‑map / B‑DoU) | 0.10 boundary F1 | For highly unbalanced masks, region losses are unstable; boundary loss integrates over the region interface and **complements** (not replaces) region losses [Boundary]. |
| *(optional)* **cbDice** | thin recall + boundary | Centerline‑Boundary Dice adds vessel‑radius weighting so the loss doesn't favor fat pads over thin traces (the "diameter imbalance" problem ≈ pads vs traces here) and adds boundary awareness [cbDice]. |
| *(optional)* **Skeleton Recall Loss** | thin recall | Recent, efficient alternative/complement to clDice for centerline recall [SkeletonRecall]. |

> **Decision:** Baseline `= Dice + soft‑clDice`. Add `+ Tversky(FN‑weighted)` for the recall terms and a small `+ Boundary` for the 0.10 term. Treat cbDice / Skeleton‑Recall as ablation upgrades. **Weight the loss to mirror the score weights** (region ≈ 0.45, centerline/topology ≈ 0.45, boundary ≈ 0.10) and tune on CV.

### 3.5 Training strategy for 260 synthetic samples
- **Cross‑validation:** 5‑fold CV; train one model per fold and **ensemble the 5** at inference. With 260 images this both regularizes and gives a held‑out score estimate that tracks the leaderboard.
- **Augmentation (albumentations):** the data is synthetic and geometry‑rich, so augment hard but **label‑preservingly** — random flips/90° rotations, small affine (scale/translate/rotate), **elastic/grid distortion** (great for curvilinear robustness), and photometric jitter (brightness/contrast/hue on top, intensity/gamma on x‑ray, mild noise/blur). **Crucially, apply the identical geometric transform to top, x‑ray, seed, and mask** so co‑registration is preserved. Avoid augmentations that would break the seed↔net relationship.
- **Regularization:** dropout in the decoder, weight decay, early stopping on the CV‑fold composite metric, EMA of weights.
- **Optimizer/schedule:** AdamW + cosine (or one‑cycle) LR; mixed precision (AMP) to speed up and cut VRAM.
- **Class imbalance at the batch level:** the loss handles it; optionally oversample crops where the net is very small.
- **Pretraining choice:** ImageNet‑pretrained encoder is the verified, simplest transfer; the data‑use rules forbid external PCB data, but ImageNet‑pretrained backbones are standard CV libraries (the rules forbid external *PCB images/datasets and answer‑key reconstruction*, not ImageNet‑pretrained generic encoders — confirm against the exact rules before submitting, see §6).

### 3.6 Inference: TTA + ensembling
- **Test‑time augmentation:** average probabilities over flips/90° rotations (un‑transform each before averaging). Cheap, reliably lifts Dice and recall.
- **Ensemble** the 5 CV folds (and optionally different backbones/loss mixes) by **averaging soft probability maps** *before* thresholding/post‑processing.
- Keep everything **forward‑pass‑only** — no inference‑time optimization, no external API at inference (a hard rule).

### 3.7 Connectivity‑aware post‑processing (this is where the 0.20 + 0.25 terms are won)
Because **100% of ground‑truth nets are a single 8‑connected component** and **the seed is always inside the net**, deterministic post‑processing is extremely effective. Recommended chain (each step verified or directly justified by the data):

1. **Hysteresis / dual‑threshold** on the probability map `p`: a **high threshold** seeds confident core pixels, a **low threshold** grows along faint thin traces. This recovers low‑confidence thin routes (helps the 0.25 recall term) without flooding background — the canonical road/vessel trick for connectivity [RoadConnectivity].
2. **Keep only the connected component that contains the seed** (8‑connectivity). This single step enforces the "one seed‑connected net" requirement and **deletes all foreign nets/blobs** → large, free gain on the 0.20 term.
3. **Gentle morphological closing** (small kernel) to **bridge sub‑pixel gaps** in thin traces and across vias *before* component selection, so a hairline break doesn't split the net. Keep the kernel small to avoid merging a *neighboring* net (which would then be wrongly kept by step 2).
4. **(Optional) Seed‑driven Fast‑Marching / geodesic region growing** using `p` as the speed image: a monotonic, positive‑speed front grows **one connected region outward from the seed by construction**, naturally matching the seed‑prompt requirement [FastMarching][SimpleITK]. Use this as an alternative/refinement to steps 1–2 when hysteresis alone leaves gaps.
5. **Size‑factor guard:** the metric caps connectivity credit at `min(1, 2·A_target/A_component)`, so an over‑grown blob is penalized. After growing, **stop expansion** and prefer the tighter mask if the component balloons well past the expected net area (≈3.5% of the image). This trades a little recall for avoiding the size penalty.

> **Caution from the research:** seeded **graph‑cut / GrabCut** has a known **"shrinking bias"** that shortcuts thin protrusions — risky for thin‑trace recall — and those specific claims were *unverified* (transient verifier failures), so **prefer connected‑component selection + Fast‑Marching/geodesic growing** over graph‑cut. Random‑walker is plausible but also unverified here; don't make it the primary mechanism.

### 3.8 Submission encoding (RLE) — correctness details
- Output mask must be exactly **192×192**, binary.
- RLE uses **1‑indexed, row‑major (C‑order) flattened positions**, written as `start length` pairs separated by spaces. **Empty mask → empty string.**
- Emit **exactly one row per `image_id` in `test.csv`**, columns exactly `image_id,mask_rle`, **no duplicates, no extra ids**.
- **Validate the encoder with a round‑trip unit test**: `decode(encode(M)) == M` on training masks, and confirm the row‑major + 1‑indexing convention against the provided `sample_submission.csv` before trusting any leaderboard number. (RLE off‑by‑one/orientation bugs are the most common silent score‑killers.)

---

## 4. How each design choice maps back to the score
| Metric term (weight) | Primary levers |
|---|---|
| Net Dice (0.45) | Pretrained encoder–decoder, early multi‑modal fusion, Dice/Tversky loss, TTA + fold ensembling. |
| Thin‑trace recall (0.25) | Native 192 resolution, soft‑clDice + FN‑weighted Tversky, hysteresis low‑threshold growth, no aggressive erosion. |
| Endpoint connectivity (0.20) | soft‑clDice training, **keep‑seed‑component**, morphological gap‑bridging, Fast‑Marching grow, size‑factor guard. |
| Boundary F1 (0.10) | Boundary/cbDice loss, accurate native‑res decoder, avoid over‑smoothing. |

---

## 5. Phased implementation plan (ship something scoring early, then climb)
1. **Phase 0 — plumbing & honest CV (½ day).** Data loaders for `[top⊕xray⊕seed]`, RLE encode/decode with a **round‑trip test**, a local re‑implementation of the **exact composite metric** for offline validation, 5‑fold split.
2. **Phase 1 — strong baseline.** SMP U‑Net, `resnet34`, early fusion, `Dice + soft‑clDice` loss, basic augmentation. Add **keep‑seed‑component** post‑processing. *This alone should score respectably.*
3. **Phase 2 — metric‑targeted gains.** Add FN‑weighted **Tversky** + **Boundary** loss; add **hysteresis dual‑threshold** + **morphological gap‑bridge** + **size‑factor guard**; add **TTA**. Ablate each against the local metric.
4. **Phase 3 — capacity & fusion.** Try `efficientnet‑b2` / `mit_b2`, **U‑Net++**, and the **first‑block‑split dual encoder** (small x‑ray stream). Add **cbDice / Skeleton‑Recall**. Keep only what improves CV.
5. **Phase 4 — ensemble & polish.** Ensemble folds + best 2–3 backbones (probability averaging), optional **Fast‑Marching** refinement, final RLE sanity checks, submission.

Expected ordering of impact: **keep‑seed‑component post‑processing** and **clDice/Tversky losses** give the biggest early jumps (they hit 0.55 of the score directly); fusion/architecture tuning and ensembling are the diminishing‑returns top end.

---

## 6. Pitfalls, refuted claims, and rules to respect
- **Don't downsample** — thin traces (~4.3 px) vanish and the 0.25 term collapses.
- **Don't use plain BCE** — 3.5% foreground makes it precision‑biased; use region/topology losses.
- **Don't trust these (research‑refuted) claims:** soft‑clDice *guarantees* topology (it doesn't — it only improves it); PTSAM "2,048 params matches SOTA"; D‑LinkNet "1st place DeepGlobe 2018"; IU‑Net's headline single‑click gains. Use the *mechanisms*, not the marketing.
- **Graph‑cut/GrabCut shrinking bias** — avoid as the primary connectivity mechanism; prefer component‑selection + Fast‑Marching.
- **RLE convention bugs** (1‑indexing, row‑major, 192×192) — unit‑test the round‑trip; the metric is unforgiving of orientation/offset errors.
- **Respect the data‑use rules:** public challenge data only; **no external PCB images/datasets**, no answer‑key reconstruction, no id/hash/template lookup exploits, **no external API at inference**, don't modify `grade.py`. Re‑read the "Data Use And Solution Constraints" section before submitting and confirm that ImageNet‑pretrained generic encoders are acceptable (they are standard public CV libraries; the prohibition is on external *PCB/board* visual assets and answer reconstruction).
- **Synthetic‑to‑test gap & crossing nets:** the hardest residual problem (separating nets that visually cross) is *not* directly solved by any vessel/road benchmark — this is where seed conditioning + connectivity post‑processing + the x‑ray's via evidence must carry the weight. Budget ablation time here.

---

## References
Verified across two adversarial literature‑review passes (papers + repos). Confidence noted where the underlying claim was a split vote.

**Promptable / interactive segmentation**
- **DEXTR** — Maninis et al., *Deep Extreme Cut*, CVPR 2018. https://arxiv.org/abs/1711.09081
- **RITM** — Sofiiuk et al., *Reviving Iterative Training with Mask Guidance*. Repo: https://github.com/saic-vul/ritm_interactive_segmentation
- **SimpleClick** — Liu et al., ICCV 2023. https://ar5iv.labs.arxiv.org/html/2210.11006 · https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_SimpleClick_Interactive_Image_Segmentation_with_Simple_Vision_Transformers_ICCV_2023_paper.pdf
- **IU‑Net** — interactive U‑Net with click channels. https://arxiv.org/pdf/2111.09740 *(headline single‑click gains refuted)*
- **FocalClick / low‑latency interactive seg** — CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Rethinking_Interactive_Image_Segmentation_with_Low_Latency_High_Quality_and_CVPR_2024_paper.pdf

**Thin‑structure / topology / imbalance losses**
- **clDice** — Shit, Paetzold et al., *clDice: A Topology‑Preserving Loss for Tubular Structure Segmentation*, CVPR 2021. https://arxiv.org/abs/2003.07311 · Repo: https://github.com/jocpae/clDice *(connectivity gains: confirmed; "homotopy guarantee": refuted)*
- **Tversky loss** — Salehi et al. https://arxiv.org/abs/1706.05721
- **Boundary loss** — Kervadec et al. https://arxiv.org/abs/1812.07032
- **cbDice (Centerline Boundary Dice)** — https://arxiv.org/abs/2407.01517
- **Skeleton Recall Loss** — ECCV 2024. https://arxiv.org/pdf/2404.10506
- **Topology‑preserving deep image segmentation** — Hu et al., NeurIPS 2019. http://papers.neurips.cc/paper/8803-topology-preserving-deep-image-segmentation.pdf
- **D‑LinkNet** — road extraction encoder–decoder. https://ieeexplore.ieee.org/document/8575492/ *(architecture/pretraining: confirmed; "DeepGlobe 1st place": refuted)*
- **Road connectivity via joint orientation+segmentation** — Batra et al., CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.pdf

**Multi‑modal fusion**
- **Fusion survey** — Zhou, Ruan & Canu, *A review: deep learning for medical image segmentation using multi‑modality fusion*, Array (Elsevier) 2020. https://arxiv.org/pdf/2004.10664
- **FuseNet** — Hazirbas et al., ACCV 2016 (dual‑encoder RGB‑D). https://link.springer.com/chapter/10.1007/978-3-319-54181-5_14
- **RSFNet** — asymmetric RGB‑Thermal fusion, Neurocomputing 2024. https://arxiv.org/pdf/2306.10364
- **Small‑data dual‑encoder vs concat** — cervical mpMRI, MDPI Bioengineering 2023. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10647438/
- **PIF‑Net + MSFF (pixel+feature fusion additive)** — Front. Neurosci. 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9515796/

**Foundation models on tiny data**
- **SAM medical evaluation** (zero‑shot fails on vessels; fine‑tune helps) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10252742/
- **PTSAM** — prompt‑tuning SAM on tiny data, CVPR‑W 2025. https://arxiv.org/pdf/2504.16739 *("2,048‑param matches SOTA": refuted)*

**Connectivity post‑processing**
- **Fast Marching Method** — Sethian, PNAS 1996. https://www.pnas.org/doi/10.1073/pnas.93.4.1591 · Wikipedia: https://en.wikipedia.org/wiki/Fast_marching_method · SimpleITK FMM docs: https://simpleitk.readthedocs.io/en/master/link_FastMarchingSegmentation_docs.html
- *(Seeded graph‑cut / random‑walker references were investigated but their specific claims were unverified due to transient failures; treat as secondary.)*

**PCB‑specific reverse‑engineering (context, applied)**
- Wire segmentation for PCB via deep CNN + graph cut — IET Image Processing. https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2017.1208
- PCB X‑ray / layer connectivity & net tracing — https://www.nature.com/articles/s41598-024-84635-2 · https://www.mdpi.com/2079-9292/13/12/2353

**Tooling**
- **segmentation‑models‑pytorch** — https://segmentation-models-pytorch.readthedocs.io/en/latest/models.html
- **MONAI** (Dice/Tversky/clDice‑style losses), **albumentations** (augmentation), **SimpleITK/scikit‑image/OpenCV** (post‑processing).
