import os, json, math, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings("ignore")

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./dataset/public"))
OUT_DIR   = Path(os.environ.get("OUT_DIR", "./working"))
BACKBONE  = os.environ.get("BACKBONE", "resnet34")
N_FOLDS   = int(os.environ.get("N_FOLDS", "5"))
EPOCHS    = int(os.environ.get("EPOCHS", "60"))
IMG_SIZE  = int(os.environ.get("IMG_SIZE", "192"))
BATCH     = int(os.environ.get("BATCH", "16"))
LR        = float(os.environ.get("LR", "3e-4"))
WD        = float(os.environ.get("WD", "1e-2"))
TTA       = os.environ.get("TTA", "1") == "1"
FAST      = os.environ.get("FAST", "0") == "1"
SEED      = int(os.environ.get("SEED", "42"))
PRETRAINED= os.environ.get("PRETRAINED", "1") == "1"
PRETRAINED_PATH = os.environ.get("PRETRAINED_PATH", "")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
W_BCE     = float(os.environ.get("W_BCE", "0.5"))
W_DICE    = float(os.environ.get("W_DICE", "1.0"))
W_TVERSKY = float(os.environ.get("W_TVERSKY", "0.5"))
W_CLDICE  = float(os.environ.get("W_CLDICE", "0.5"))
TV_ALPHA  = float(os.environ.get("TV_ALPHA", "0.3"))
TV_BETA   = float(os.environ.get("TV_BETA", "0.7"))
HI_THR    = float(os.environ.get("HI_THR", "0.55"))
LO_THR    = float(os.environ.get("LO_THR", "0.35"))
CLOSE_K   = int(os.environ.get("CLOSE_K", "3"))
SEED_SIGMA= float(os.environ.get("SEED_SIGMA", "5.0"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    AMP_DTYPE = torch.float32
USE_SCALER = DEVICE == "cuda" and AMP_DTYPE == torch.float16
IN_CHANS = 6
HW = (IMG_SIZE, IMG_SIZE)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if FAST:
    N_FOLDS, EPOCHS = 1, 1

OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(SEED)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def rle_encode(mask):
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="C")
    if pixels.sum() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(rle, shape=HW):
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        arr = np.array(rle.split(), dtype=int)
        starts = arr[0::2] - 1
        lengths = arr[1::2]
        for s, l in zip(starts, lengths):
            flat[s:s + l] = 1
    return flat.reshape(shape, order="C")


def read_gray(path):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(str(path))
    if a.shape[:2] != HW:
        a = cv2.resize(a, HW, interpolation=cv2.INTER_NEAREST)
    return a


def read_rgb(path):
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(str(path))
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    if a.shape[:2] != HW:
        a = cv2.resize(a, HW, interpolation=cv2.INTER_LINEAR)
    return a


def build_seed_channels(seed_bin):
    disk = (seed_bin > 0).astype(np.float32)
    g = cv2.GaussianBlur(disk, (0, 0), SEED_SIGMA)
    m = g.max()
    if m > 1e-6:
        g = g / m
    return disk, g.astype(np.float32)


def build_input(top_rgb, xray_rgb, seed_bin):
    top = top_rgb.astype(np.float32) / 255.0
    top = (top - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    xray = cv2.cvtColor(xray_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    xray = (xray - 0.5) / 0.25
    disk, gauss = build_seed_channels(seed_bin)
    x = np.stack([top[..., 0], top[..., 1], top[..., 2], xray, disk, gauss], axis=2)
    return x.astype(np.float32)


RAW_CACHE = {}


def get_raw(r):
    iid = r["image_id"]
    cached = RAW_CACHE.get(iid)
    if cached is not None:
        return cached
    top = read_rgb(DATA_ROOT / r["top_path"])
    xray = read_rgb(DATA_ROOT / r["xray_path"])
    seed = read_gray(DATA_ROOT / r["seed_mask_path"])
    mask = read_gray(DATA_ROOT / r["mask_path"]) if "mask_path" in r.index else None
    RAW_CACHE[iid] = (top, xray, seed, mask)
    return RAW_CACHE[iid]


def d4_apply(arr, op):
    if op == 0:
        return arr
    if op == 1:
        return arr[::-1, ...]
    if op == 2:
        return arr[:, ::-1, ...]
    if op == 3:
        return arr[::-1, ::-1, ...]
    if op == 4:
        return np.rot90(arr, 1)
    if op == 5:
        return np.rot90(arr, 3)
    return arr


def affine_warp(img6, mask, rot, scale, tx, ty):
    c = (IMG_SIZE / 2.0, IMG_SIZE / 2.0)
    M = cv2.getRotationMatrix2D(c, rot, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = np.empty_like(img6)
    for k in range(img6.shape[2]):
        out[..., k] = cv2.warpAffine(img6[..., k], M, HW,
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mout = cv2.warpAffine(mask, M, HW, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, mout


def elastic(img6, mask, alpha, sigma):
    dx = cv2.GaussianBlur((np.random.rand(*HW).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*HW).astype(np.float32) * 2 - 1), (0, 0), sigma) * alpha
    gx, gy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    mx = (gx + dx).astype(np.float32)
    my = (gy + dy).astype(np.float32)
    out = np.empty_like(img6)
    for k in range(img6.shape[2]):
        out[..., k] = cv2.remap(img6[..., k], mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mout = cv2.remap(mask, mx, my, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, mout


def photometric(img6):
    out = img6.copy()
    if random.random() < 0.5:
        g = 1.0 + (random.random() - 0.5) * 0.4
        out[..., 0:4] = out[..., 0:4] * g
    if random.random() < 0.5:
        b = (random.random() - 0.5) * 0.3
        out[..., 0:4] = out[..., 0:4] + b
    if random.random() < 0.3:
        out[..., 0:4] = out[..., 0:4] + np.random.randn(IMG_SIZE, IMG_SIZE, 4).astype(np.float32) * 0.03
    return out


def augment(img6, mask):
    op = random.randint(0, 5)
    img6 = np.ascontiguousarray(d4_apply(img6, op))
    mask = np.ascontiguousarray(d4_apply(mask, op))
    if random.random() < 0.8:
        rot = (random.random() - 0.5) * 30.0
        scale = 1.0 + (random.random() - 0.5) * 0.3
        tx = (random.random() - 0.5) * 16.0
        ty = (random.random() - 0.5) * 16.0
        img6, mask = affine_warp(img6, mask, rot, scale, tx, ty)
    if random.random() < 0.15:
        img6, mask = elastic(img6, mask, alpha=10.0, sigma=4.0)
    img6 = photometric(img6)
    return img6, mask


class CircuitDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.has_y = "mask_path" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        top, xray, seed, mask_raw = get_raw(r)
        img6 = build_input(top, xray, seed)
        if self.has_y and mask_raw is not None:
            mask = (mask_raw > 127).astype(np.float32)
        else:
            mask = np.zeros(HW, dtype=np.float32)
        if self.train:
            img6, mask = augment(img6, mask)
        x = torch.from_numpy(np.ascontiguousarray(img6.transpose(2, 0, 1)))
        y = torch.from_numpy(np.ascontiguousarray(mask))[None, ...]
        s = torch.from_numpy(np.ascontiguousarray((seed > 0).astype(np.float32)))[None, ...]
        return {"x": x, "y": y, "seed": s, "id": r["image_id"]}


import timm


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class UNetDecoder(nn.Module):
    def __init__(self, enc_channels, dec_channels=(256, 128, 64, 32)):
        super().__init__()
        enc = list(enc_channels[::-1])
        self.blocks = nn.ModuleList()
        in_ch = enc[0]
        for i, dc in enumerate(dec_channels):
            skip_ch = enc[i + 1]
            self.blocks.append(DecoderBlock(in_ch, skip_ch, dc))
            in_ch = dc
        self.out_ch = in_ch

    def forward(self, feats):
        feats = feats[::-1]
        x = feats[0]
        for i, blk in enumerate(self.blocks):
            x = blk(x, feats[i + 1])
        return x


class SegModel(nn.Module):
    def __init__(self, backbone, in_chans, pretrained):
        super().__init__()
        kw = dict(features_only=True, in_chans=in_chans, out_indices=(0, 1, 2, 3, 4))
        if pretrained and PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
            self.encoder = timm.create_model(backbone, pretrained=True,
                                             pretrained_cfg_overlay=dict(file=PRETRAINED_PATH), **kw)
        else:
            self.encoder = timm.create_model(backbone, pretrained=pretrained, **kw)
        ch = self.encoder.feature_info.channels()
        self.decoder = UNetDecoder(ch)
        self.head = nn.Conv2d(self.decoder.out_ch, 1, 1)

    def forward(self, x):
        size = x.shape[-2:]
        feats = self.encoder(x)
        d = self.decoder(feats)
        d = F.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.head(d)


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, (3, 3), 1, (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(prob, target, iters=10, smooth=1.0):
    sk_p = soft_skel(prob, iters)
    sk_t = soft_skel(target, iters)
    tprec = (torch.sum(sk_p * target) + smooth) / (torch.sum(sk_p) + smooth)
    tsens = (torch.sum(sk_t * prob) + smooth) / (torch.sum(sk_t) + smooth)
    cl = 2.0 * tprec * tsens / (tprec + tsens)
    return 1.0 - cl


def dice_loss(prob, target, smooth=1.0):
    num = 2 * (prob * target).sum(dim=(1, 2, 3)) + smooth
    den = (prob + target).sum(dim=(1, 2, 3)) + smooth
    return (1 - num / den).mean()


def tversky_loss(prob, target, alpha, beta, smooth=1.0):
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    t = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - t).mean()


bce_logits = nn.BCEWithLogitsLoss()


def compute_loss(logits, target, cldice_scale):
    prob = torch.sigmoid(logits)
    loss = W_BCE * bce_logits(logits, target)
    loss = loss + W_DICE * dice_loss(prob, target)
    if W_TVERSKY > 0:
        loss = loss + W_TVERSKY * tversky_loss(prob, target, TV_ALPHA, TV_BETA)
    if W_CLDICE > 0 and cldice_scale > 0:
        loss = loss + W_CLDICE * cldice_scale * soft_cldice_loss(prob, target)
    return loss


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


STRUCT8 = np.ones((3, 3), dtype=int)


def postprocess(prob, seed_bin):
    from scipy import ndimage
    from skimage.filters import apply_hysteresis_threshold
    seed_bin = seed_bin > 0
    if prob.max() < LO_THR:
        mask = seed_bin.copy()
    else:
        mask = apply_hysteresis_threshold(prob, LO_THR, HI_THR)
        mask = mask | seed_bin
    if CLOSE_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lbl, n = ndimage.label(mask, structure=STRUCT8)
    if n > 0:
        seed_lbls = set(np.unique(lbl[seed_bin]).tolist()) - {0}
        if seed_lbls:
            mask = np.isin(lbl, list(seed_lbls))
        else:
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
            biggest = int(np.argmax(sizes)) + 1
            mask = lbl == biggest
    return mask.astype(np.uint8)


def boundary_pixels(mask):
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8))
    return (m - er).astype(bool)


def metric_dice(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2 * tp / s)


def metric_thin_recall(pred, gt):
    from skimage.morphology import skeletonize
    if gt.sum() == 0:
        return 1.0
    sk = skeletonize(gt > 0)
    denom = sk.sum()
    if denom == 0:
        return 1.0
    return float(np.logical_and(sk, pred > 0).sum() / denom)


def metric_connectivity(pred, gt, seed_bin):
    from scipy import ndimage
    seed_bin = seed_bin > 0
    if gt.sum() == 0:
        return 1.0 if pred.sum() == 0 else 0.0
    lbl, n = ndimage.label(pred, structure=STRUCT8)
    comp = np.zeros_like(pred, dtype=bool)
    seed_lbls = set(np.unique(lbl[seed_bin]).tolist()) - {0}
    if seed_lbls:
        comp = np.isin(lbl, list(seed_lbls))
    cover = np.logical_and(comp, gt).sum() / max(1, gt.sum())
    a_comp = comp.sum()
    a_tgt = gt.sum()
    size_factor = min(1.0, (2.0 * a_tgt) / max(1, a_comp))
    return float(cover * size_factor)


def metric_boundary_f1(pred, gt, tol=2):
    bp = boundary_pixels(pred)
    bg = boundary_pixels(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dt_g = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 3)
    prec = (dt_g[bp] <= tol).mean()
    rec = (dt_p[bg] <= tol).mean()
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def challenge_score(pred, gt, seed_bin):
    d = metric_dice(pred, gt)
    tr = metric_thin_recall(pred, gt)
    ec = metric_connectivity(pred, gt, seed_bin)
    bf = metric_boundary_f1(pred, gt)
    total = 0.45 * d + 0.25 * tr + 0.20 * ec + 0.10 * bf
    return total, d, tr, ec, bf


def make_loader(df, train, batch):
    ds = CircuitDataset(df, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=NUM_WORKERS,
                      pin_memory=(DEVICE == "cuda"), drop_last=train)


@torch.no_grad()
def predict_probs(model, df, batch):
    model.eval()
    dl = make_loader(df, False, batch)
    out = {}
    tta_ops = [0, 1, 2, 3] if TTA else [0]
    for b in dl:
        x = b["x"].to(DEVICE)
        acc = torch.zeros((x.shape[0], 1, IMG_SIZE, IMG_SIZE), device=DEVICE)
        for op in tta_ops:
            xi = x
            if op == 1:
                xi = torch.flip(x, dims=[2])
            elif op == 2:
                xi = torch.flip(x, dims=[3])
            elif op == 3:
                xi = torch.flip(x, dims=[2, 3])
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    p = torch.sigmoid(model(xi).float())
            else:
                p = torch.sigmoid(model(xi))
            if op == 1:
                p = torch.flip(p, dims=[2])
            elif op == 2:
                p = torch.flip(p, dims=[3])
            elif op == 3:
                p = torch.flip(p, dims=[2, 3])
            acc = acc + p
        acc = acc / len(tta_ops)
        acc = acc.squeeze(1).cpu().numpy()
        for i, iid in enumerate(b["id"]):
            out[iid] = acc[i]
    return out


def train_fold(tr_df, va_df, tag):
    model = SegModel(BACKBONE, IN_CHANS, PRETRAINED).to(DEVICE)
    enc_params = list(model.encoder.parameters())
    dec_params = list(model.decoder.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": LR * 0.3},
        {"params": dec_params, "lr": LR},
    ], weight_decay=WD)
    dl = make_loader(tr_df, True, BATCH)
    steps = max(1, len(dl) * EPOCHS)
    warmup = max(1, int(0.05 * steps))

    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)
    ema = EMA(model, EMA_DECAY)
    step = 0
    for ep in range(EPOCHS):
        model.train()
        cldice_scale = min(1.0, (ep + 1) / max(1, math.ceil(EPOCHS * 0.5)))
        last = 0.0
        for b in dl:
            x = b["x"].to(DEVICE, non_blocking=True)
            y = b["y"].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE):
                    logits = model(x)
                    loss = compute_loss(logits.float(), y, cldice_scale)
                if USE_SCALER:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            else:
                logits = model(x)
                loss = compute_loss(logits, y, cldice_scale)
                loss.backward()
                opt.step()
            sched.step()
            ema.update(model)
            last = float(loss.item())
            step += 1
        log(f"  {tag} epoch {ep + 1}/{EPOCHS} loss {last:.4f}")
    eval_model = SegModel(BACKBONE, IN_CHANS, False).to(DEVICE)
    eval_model.load_state_dict(ema.shadow, strict=True)
    eval_model.eval()
    return eval_model


def oof_metrics(prob_map, va_df):
    rows = []
    for _, r in va_df.iterrows():
        iid = r["image_id"]
        seed = read_gray(DATA_ROOT / r["seed_mask_path"])
        gt = (read_gray(DATA_ROOT / r["mask_path"]) > 127).astype(np.uint8)
        pred = postprocess(prob_map[iid], seed)
        sc = challenge_score(pred, gt, seed)
        rows.append(sc)
    arr = np.array(rows, dtype=np.float32)
    return arr.mean(axis=0)


def main():
    log(f"config backbone={BACKBONE} folds={N_FOLDS} epochs={EPOCHS} img={IMG_SIZE} "
        f"batch={BATCH} pretrained={PRETRAINED} fast={FAST} device={DEVICE}")
    tr = pd.read_csv(DATA_ROOT / "train.csv")
    te = pd.read_csv(DATA_ROOT / "test.csv")
    if FAST:
        tr = tr.head(24).reset_index(drop=True)
        te = te.head(8).reset_index(drop=True)
    log(f"train {tr.shape} test {te.shape}")

    for _, r in tr.iterrows():
        get_raw(r)
    for _, r in te.iterrows():
        get_raw(r)
    log(f"cached {len(RAW_CACHE)} raw images")

    test_seed = {}
    for _, r in te.iterrows():
        test_seed[r["image_id"]] = read_gray(DATA_ROOT / r["seed_mask_path"])

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=max(2, N_FOLDS), shuffle=True, random_state=SEED)
    folds = list(kf.split(tr))
    if FAST:
        folds = folds[:1]

    test_prob_acc = {iid: np.zeros(HW, dtype=np.float32) for iid in te["image_id"]}
    n_used = 0
    oof_acc = []

    def finalize():
        rows = []
        for iid in te["image_id"]:
            prob = test_prob_acc[iid] / max(1, n_used)
            pred = postprocess(prob, test_seed[iid])
            rows.append({"image_id": iid, "mask_rle": rle_encode(pred)})
        sub = pd.DataFrame(rows, columns=["image_id", "mask_rle"])
        sub.to_csv(OUT_DIR / "submission.csv", index=False)
        metrics = {"folds_done": n_used}
        if oof_acc:
            mean_oof = np.mean(np.stack(oof_acc, axis=0), axis=0)
            metrics.update({
                "oof_score": float(mean_oof[0]), "oof_dice": float(mean_oof[1]),
                "oof_thin_recall": float(mean_oof[2]), "oof_connectivity": float(mean_oof[3]),
                "oof_boundary_f1": float(mean_oof[4])})
        metrics["config"] = {"backbone": BACKBONE, "folds": N_FOLDS, "epochs": EPOCHS,
                             "img": IMG_SIZE, "pretrained": PRETRAINED}
        json.dump(metrics, open(OUT_DIR / "metrics.json", "w"), indent=2)
        log(f"  wrote submission {sub.shape} metrics {metrics}")

    for fi, (tri, vai) in enumerate(folds[:N_FOLDS]):
        t0 = time.time()
        tr_df, va_df = tr.iloc[tri], tr.iloc[vai]
        log(f"=== fold {fi + 1}/{N_FOLDS} train {len(tr_df)} val {len(va_df)} ===")
        model = train_fold(tr_df, va_df, tag=f"f{fi}")
        va_prob = predict_probs(model, va_df, BATCH * 2)
        m = oof_metrics(va_prob, va_df)
        oof_acc.append(m)
        log(f"  fold {fi + 1} OOF score {m[0]:.4f} dice {m[1]:.4f} thin {m[2]:.4f} "
            f"conn {m[3]:.4f} bf1 {m[4]:.4f} ({time.time() - t0:.0f}s)")
        te_prob = predict_probs(model, te, BATCH * 2)
        for iid in te["image_id"]:
            test_prob_acc[iid] += te_prob[iid]
        n_used += 1
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        finalize()

    log("DONE")


if __name__ == "__main__":
    main()
