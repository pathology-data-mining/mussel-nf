"""Generate synthetic fixtures for the ABMIL benchmark integration test."""
import os, h5py, numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

N_SLIDES   = 20
N_FEATURES = 8
RNG        = np.random.default_rng(42)

SLIDE_IDS  = [f"abmil_slide_{i:04d}" for i in range(1, N_SLIDES + 1)]
# Binary labels: first 10 positive, last 10 negative.
LABELS     = [1] * (N_SLIDES // 2) + [0] * (N_SLIDES // 2)

for slide_id in SLIDE_IDS:
    n_tiles = int(RNG.integers(5, 20))
    features = RNG.standard_normal((n_tiles, N_FEATURES)).astype("float32")
    with h5py.File(os.path.join(OUT, f"{slide_id}.h5"), "w") as f:
        f.create_dataset("features", data=features)

df = pd.DataFrame({"slide_id": SLIDE_IDS, "label": LABELS})
df.to_parquet(os.path.join(OUT, "labels.parquet"), index=False)
print(f"Generated {N_SLIDES} slides + labels.parquet in {OUT}")
