# Slide-Level Model Support

Slide-level models aggregate patch-level features into a single slide-level
representation. Specify a slide encoder in `featurize.model_types` — the
required patch encoder is resolved automatically via
`params.featurize.slide_to_patch_mapping`.

## Supported models

| Slide encoder key | Model | Auto-selected patch encoder |
|---|---|---|
| `gigapath_slide` | Prov-GigaPath | `gigapath` |
| `titan_slide` | TITAN | `conch1_5` |
| `prism_slide` | PRISM | `virchow` |
| `feather_slide` | CONCH/FEATHER | `conch1_5` |
| `chief_slide` | CHIEF | `ctranspath` |
| `madeleine_slide` | MADELEINE | `clip` |
| `abmil_slide` | ABMIL (encoder-agnostic) | specify patch encoder separately |

## Usage

```bash
nextflow run main.nf --samples_csv slides.csv \
  --featurize.model_types='[gigapath_slide]'
```

Or in a params file:

```yaml
featurize:
  model_types: [gigapath_slide]
  slide_batch_size: 8   # slides per slide-encoder forward pass
```

Multiple slide models run in parallel:

```yaml
featurize:
  model_types: [gigapath_slide, titan_slide]
```

## Custom model paths

```yaml
featurize:
  model_types: [gigapath_slide]
  model_paths:
    gigapath: /path/to/gigapath.pth
  slide_model_paths:
    gigapath_slide: /path/to/gigapath_slide_model
```

## Output structure

```
results/
  features/
    gigapath_slide/       # slide-level features
      *.features.pt
      *.features.h5
    gigapath/             # intermediate patch-level features
      *.patch_features.pt
      *.patch_features.h5
  tiles/
    *.patch.h5            # tile coordinates
```

## Batching parameters

| Parameter | Default | Description |
|---|---|---|
| `featurize.workflow_batch_size` | 8 | Slides per Nextflow task |
| `featurize.batch_size` | 64 | Tiles per patch-encoder forward pass |
| `featurize.slide_batch_size` | 8 | Slides per slide-encoder forward pass |

## Notes

- Models download from HuggingFace automatically; set the `HF_TOKEN` Nextflow
  secret for gated models: `nextflow secrets set HF_TOKEN <token>`
- `abmil_slide` is encoder-agnostic and not in `slide_to_patch_mapping`; pair
  it explicitly with a patch encoder in `model_types`.
