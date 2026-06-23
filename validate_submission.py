import sys
import numpy as np
import pandas as pd
from pathlib import Path

H, W = 192, 192


def rle_decode(rle, shape=(H, W)):
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        arr = np.array(rle.split(), dtype=int)
        starts = arr[0::2] - 1
        lengths = arr[1::2]
        for s, l in zip(starts, lengths):
            flat[s:s + l] = 1
    return flat.reshape(shape, order="C")


def main(sub_path="./working/submission.csv", test_path="./dataset/public/test.csv"):
    sub = pd.read_csv(sub_path, dtype={"image_id": str, "mask_rle": str}).fillna({"mask_rle": ""})
    test = pd.read_csv(test_path, dtype={"image_id": str})

    assert list(sub.columns) == ["image_id", "mask_rle"], f"columns must be image_id,mask_rle got {list(sub.columns)}"
    assert sub["image_id"].is_unique, "duplicate image_id"
    assert set(sub["image_id"]) == set(test["image_id"]), "image_id set does not match test.csv exactly"
    assert len(sub) == len(test), f"row count {len(sub)} != test {len(test)}"

    total_fg = 0
    for _, row in sub.iterrows():
        rle = row["mask_rle"]
        if not isinstance(rle, str) or rle.strip() == "":
            continue
        toks = rle.split()
        assert len(toks) % 2 == 0, f"odd token count for {row['image_id']}"
        arr = np.array(toks, dtype=int)
        starts = arr[0::2]
        lengths = arr[1::2]
        assert (starts >= 1).all(), f"non positive start index for {row['image_id']}"
        assert (lengths >= 1).all(), f"non positive run length for {row['image_id']}"
        ends = starts + lengths - 1
        assert (ends <= H * W).all(), f"run exceeds image bounds for {row['image_id']}"
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        assert np.all(s_sorted[1:] > e_sorted[:-1]), f"overlapping runs for {row['image_id']}"
        m = rle_decode(rle)
        assert m.shape == (H, W), f"decoded shape wrong for {row['image_id']}"
        total_fg += int(m.sum())

    print(f"OK rows={len(sub)} unique={sub['image_id'].nunique()} "
          f"nonempty={int((sub['mask_rle'].str.strip() != '').sum())} total_fg_px={total_fg}")
    print("Submission validation passed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
