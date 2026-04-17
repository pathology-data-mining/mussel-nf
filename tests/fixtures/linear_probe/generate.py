"""Generate synthetic fixtures for the linear probe integration test."""
import os, h5py, numpy as np, yaml
from PIL import Image

OUT = "/gpfs/mskmind_ess/limr/repos/mussel-nf/tests/fixtures/linear_probe"
os.makedirs(OUT, exist_ok=True)

N_SLIDES   = 20   # enough for stratified train/val/test with 2 classes
N_TILES    = 10
N_FEATURES = 8
PATCH_SIZE = 64
WIDTH      = N_TILES * PATCH_SIZE   # 640
HEIGHT     = PATCH_SIZE             # 64
RNG        = np.random.default_rng(42)

SLIDE_IDS  = [f"lp_slide_{i:04d}" for i in range(1, N_SLIDES + 1)]
POS_SLIDES = set(SLIDE_IDS[:N_SLIDES // 2])

for slide_id in SLIDE_IDS:
    positive = slide_id in POS_SLIDES
    # Features drawn from class-specific Gaussians so there is a weak real signal
    mean     = 0.5 if positive else -0.5
    features = RNG.normal(loc=mean, scale=1.0, size=(N_TILES, N_FEATURES)).astype("float32")
    coords   = np.array([[i * PATCH_SIZE, 0] for i in range(N_TILES)], dtype="float32")

    h5_path = os.path.join(OUT, f"{slide_id}.h5")
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset("coords", data=coords)
        ds.attrs["patch_size"] = PATCH_SIZE
        f.create_dataset("features", data=features)

    # BMP: positive slides -> left half class-1 (tumour), right half class-4 (non-tumour)
    #      negative slides -> entirely class-4
    bmp = np.full((HEIGHT, WIDTH), 4, dtype="uint8")
    if positive:
        bmp[:, : WIDTH // 2] = 1
    bmp_path = os.path.join(OUT, f"{slide_id}.bmp")
    Image.fromarray(bmp).save(bmp_path)

class_mapping = {1: 1, 4: 0}
with open(os.path.join(OUT, "class_mapping.yaml"), "w") as f:
    yaml.dump(class_mapping, f)

with open(os.path.join(OUT, "annotations.csv"), "w") as f:
    f.write("slide_id,annotation_bmp_path\n")
    for slide_id in SLIDE_IDS:
        f.write(f"{slide_id},{os.path.join(OUT, slide_id + '.bmp')}\n")

print(f"Generated {N_SLIDES} slides in {OUT}")
for fn in sorted(os.listdir(OUT)):
    sz = os.path.getsize(os.path.join(OUT, fn))
    print(f"  {fn}  ({sz} bytes)")
