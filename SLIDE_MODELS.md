# Slide-Level Model Support

This document explains how to use slide-level models (e.g., GIGAPATH_SLIDE, TITAN_SLIDE) in the mussel-nf workflow.

## Overview

Slide-level models aggregate patch-level features into a single slide-level representation using learned encoders. This provides better performance than simple pooling (mean/max) for downstream tasks.

### Supported Slide-Level Models

| Model | Key | Required Patch Encoder |
|---|---|---|
| Prov-GigaPath | `gigapath_slide` | `gigapath` |
| TITAN | `titan_slide` | `conch1_5` |
| PRISM | `prism_slide` | `virchow` |
| CONCH/FEATHER | `feather_slide` | `conch1_5` |
| CHIEF | `chief_slide` | `ctranspath` |
| MADELEINE | `madeleine_slide` | `clip` |
| ABMIL | `abmil_slide` | (encoder-agnostic) |

## Configuration

Slide encoders are specified directly in `params.featurize.model_types` — the required patch encoder is automatically resolved via `params.featurize.slide_to_patch_mapping`.

```groovy
// nextflow.config
params {
  featurize {
    model_types = ['gigapath_slide']   // patch encoder (gigapath) auto-selected
    batch_size = 64
    slide_batch_size = 8              // slides per slide-encoder forward pass
    workflow_batch_size = 8           // slides grouped per Nextflow task
  }
}
```

Or via command line / params file:
```bash
nextflow run main.nf --samples_csv slides.csv \
  --featurize.model_types='[gigapath_slide]' \
  --featurize.slide_batch_size=8
```

### Multiple slide models

```groovy
params.featurize.model_types = ['gigapath_slide', 'titan_slide']
```

### Custom model paths

```groovy
params {
  featurize {
    model_types = ['gigapath_slide']
    model_paths {
      gigapath = "/path/to/gigapath.pth"
    }
    slide_model_paths {
      gigapath_slide = "/path/to/gigapath_slide_model"
    }
  }
}
```

## Output Structure

```
results/
├── features/
│   ├── gigapath_slide/       # Slide-level features (from slide encoder)
│   │   ├── pt/
│   │   │   ├── slide1.features.pt
│   │   │   └── slide2.features.pt
│   │   └── h5/
│   │       ├── slide1.features.h5
│   │       └── slide2.features.h5
│   └── gigapath/             # Patch-level features (intermediate)
│       ├── pt/
│       │   └── ...patch_features.pt
│       └── h5/
│           └── ...patch_features.h5
├── tiles/                    # Tile coordinates (.patch.h5)
└── manifest-*.csv
```

## Model Requirements

| Slide model | Required patch encoder | Default patch size |
|---|---|---|
| `gigapath_slide` | `gigapath` | 256px |
| `titan_slide` | `conch1_5` | 512px |

Models are downloaded from HuggingFace automatically unless paths are set in
`params.featurize.model_paths` / `params.featurize.slide_model_paths`.

## Batching

Three independent batch dimensions:

| Parameter | Default | Description |
|---|---|---|
| `featurize.workflow_batch_size` | 8 | Slides per Nextflow task (scheduling overhead) |
| `featurize.batch_size` | 64 | Tiles per patch-encoder forward pass |
| `featurize.slide_batch_size` | 8 | Slides per slide-encoder forward pass |

### GPU memory guidance

| GPU VRAM | `batch_size` | `workflow_batch_size` | `slide_batch_size` |
|---|---|---|---|
| 16 GB | 32 | 4 | 4 |
| 24 GB | 64 | 8 | 8 |
| 40 GB+ | 128 | 16 | 16 |

## Troubleshooting

### Out of GPU memory
Reduce `batch_size` and/or `slide_batch_size`.

### Model download issues
Set the `HF_TOKEN` Nextflow secret: `nextflow secrets set HF_TOKEN <token>`

### Compatibility
Requires Mussel ≥ v1.1.0.

This document explains how to use slide-level models (e.g., GIGAPATH_SLIDE, TITAN_SLIDE) in the mussel-nf workflow.

## Overview

Slide-level models aggregate patch-level features into a single slide-level representation using learned encoders. This provides better performance than simple pooling (mean/max) for downstream tasks.

### Supported Slide-Level Models

- **GIGAPATH_SLIDE**: Prov-GigaPath slide encoder (requires GIGAPATH patch encoder)
- **TITAN_SLIDE**: TITAN slide encoder (requires CONCH1_5 patch encoder)

## Configuration

### Option 1: Using `tessellate_featurize` module (Recommended for most use cases)

The `tessellate_featurize` module combines tessellation and feature extraction in a single step with batching support.

```groovy
// In nextflow.config
params {
  tessellate_featurize {
    use_gpu = true
    batch_size = 64
    slide_batch_size = 8  // Process 8 slides at once for slide-level aggregation
    save_features_to_h5 = true
  }
}

// In your workflow
include { TESSELLATE_FEATURIZE_BATCH } from './modules/tessellate_featurize'

// Create model config with slide model support
// Format: [model_type, model_path, slide_model_type, slide_model_path]
ch_model_configs = Channel.of(
    ['gigapath', null, 'gigapath_slide', null]  // Uses default HuggingFace models
)

// Or with custom paths:
ch_model_configs = Channel.of(
    ['gigapath', '/path/to/gigapath.pkl', 'gigapath_slide', '/path/to/gigapath_slide']
)

// Process slides in batches
ch_samples.collate(8)  // Batch 8 slides together
    | TESSELLATE_FEATURIZE_BATCH(ch_model_configs)
```

### Option 2: Using separate `TESSELLATE` and `FEATURIZE_BATCH` processes

When you already have tessellated patches and want to batch feature extraction:

```groovy
// In nextflow.config
params {
  featurize {
    model_types = ['gigapath']
    use_gpu = true
    batch_size = 64
    slide_batch_size = 8  // For slide-level aggregation
    model_paths {
        gigapath = "/path/to/gigapath_model.pkl"
    }
    slide_model_types = ['gigapath_slide']
    slide_model_paths {
        gigapath_slide = "/path/to/gigapath_slide_model"
    }
  }
}

// In your workflow
include { TESSELLATE } from './modules/tessellation'
include { FEATURIZE_BATCH } from './modules/featurize'

// Tessellate first
ch_patches = TESSELLATE(ch_samples)

// Create model configs
ch_model_configs = Channel.fromList(params.featurize.model_types).map { model_type ->
    model_path = params.featurize.model_paths && params.featurize.model_paths[model_type] ? params.featurize.model_paths[model_type] : null
    slide_model_type = params.featurize.slide_model_types ? params.featurize.slide_model_types[0] : null
    slide_model_path = slide_model_type && params.featurize.slide_model_paths ? params.featurize.slide_model_paths[slide_model_type] : null
    [model_type, model_path, slide_model_type, slide_model_path]
}

// Batch featurize (combines slide, patch_h5)
ch_samples.combine(ch_patches.h5, by: 0)  // Join by meta
    .collate(8)  // Batch 8 slides together
    | FEATURIZE_BATCH(ch_model_configs, true)
```

### Option 3: Using separate processes without batching (Original workflow)

```groovy
// In modules/mussel.nf (already updated)
// The workflow automatically pairs patch and slide models
// No batching - processes one slide at a time
```

## Batch Processing Benefits

When using slide-level models with batch processing:

- **6-8x speedup** for multiple slides
- Model loaded only once for all slides
- GPU parallelization across slides
- Reduced memory overhead

### Example: Batch processing with GIGAPATH_SLIDE

```bash
# Process 100 slides with GIGAPATH_SLIDE aggregation
nextflow run main.nf \
    --samples_csv slides.csv \
    --outdir results \
    -profile slurm \
    -params-file gigapath_slide_config.yaml
```

**gigapath_slide_config.yaml:**
```yaml
tessellate_featurize:
  use_gpu: true
  batch_size: 128
  slide_batch_size: 8
  save_features_to_h5: true
```

## Output Structure

When using slide-level models, outputs are organized by the slide model type:

```
results/
├── features/
│   ├── gigapath_slide/           # Slide-level features
│   │   ├── h5/
│   │   │   ├── slide1.features.h5
│   │   │   └── slide2.features.h5
│   │   ├── pt/
│   │   │   ├── slide1.features.pt
│   │   │   └── slide2.features.pt
│   │   └── tile_h5/              # Patch coordinates (no features if save_features_to_h5=false)
│   │       ├── slide1.patch.h5
│   │       └── slide2.patch.h5
│   └── gigapath/                 # Patch-level features (intermediate)
│       ├── tile_h5/
│       │   ├── slide1.patch.h5   # Contains patch-level features
│       │   └── slide2.patch.h5
│       └── pt/
│           ├── slide1.features.pt
│           └── slide2.features.pt
```

## Model Requirements

### GIGAPATH_SLIDE
- Patch encoder: GIGAPATH
- Default patch size: 256px
- Automatically downloads from HuggingFace if no model_path specified

### TITAN_SLIDE
- Patch encoder: CONCH1_5  
- Default patch size: 512px
- Automatically downloads from HuggingFace if no model_path specified

## Advanced Usage

### Multiple Slide Models

Process with multiple slide-level models simultaneously:

```groovy
ch_model_configs = Channel.of(
    ['gigapath', null, 'gigapath_slide', null],
    ['conch1_5', null, 'titan_slide', null]
)
```

### Custom Aggregation Methods

The underlying CLI also supports simple pooling:

```bash
tessellate_extract_features \
    slide_path=slide.svs \
    output_h5_path=features.h5 \
    output_pt_path=features.pt \
    model_type=GIGAPATH \
    aggregation_method=mean  # or 'max'
```

## Workflow Usage Patterns

### Pattern 1: Single Slide with Slide Model (FEATURIZE)
For processing one slide at a time with slide-level aggregation:

```groovy
include { FEATURIZE } from './modules/featurize'

ch_model_configs = Channel.of(
    ['gigapath', null, 'gigapath_slide', null]
)

ch_samples.combine(ch_patches.h5, by: 0)
    | FEATURIZE(ch_model_configs, true)
```

### Pattern 2: Batch Feature Extraction (FEATURIZE_BATCH - Recommended)
For processing multiple slides with slide-level aggregation (6-8x faster):

```groovy
include { FEATURIZE_BATCH } from './modules/featurize'

ch_model_configs = Channel.of(
    ['gigapath', null, 'gigapath_slide', null]
)

ch_samples.combine(ch_patches.h5, by: 0)
    .collate(8)  // Batch 8 slides together
    | FEATURIZE_BATCH(ch_model_configs, true)
```

### Pattern 3: All-in-One Batch Processing (TESSELLATE_FEATURIZE_BATCH)
For tessellation + feature extraction in one step with batching:

```groovy
include { TESSELLATE_FEATURIZE_BATCH } from './modules/tessellate_featurize'

ch_model_configs = Channel.of(
    ['gigapath', null, 'gigapath_slide', null]
)

ch_samples.collate(8)
    | TESSELLATE_FEATURIZE_BATCH(ch_model_configs)
```

### Pattern 4: Patch-Level Only (No Slide Aggregation)
For extracting only patch-level features without slide-level aggregation:

```groovy
ch_model_configs = Channel.of(
    ['gigapath', null, null, null]  // No slide model
)
```

## Performance Tips

1. **Use batching for multiple slides**: Set `slide_batch_size=8` or higher
2. **Enable GPU**: Set `use_gpu=true` 
3. **Optimize batch_size**: Larger values (128-256) for patch extraction
4. **Disable H5 features**: Set `save_features_to_h5=false` to save only PT files (saves disk space)

## Troubleshooting

### Out of GPU Memory
- Reduce `batch_size` (try 64 or 32)
- Reduce `slide_batch_size` (try 4 or 2)

### Model Download Issues
- Set `HF_TOKEN` environment variable for gated models
- Pre-download models and specify `model_path`

### Compatibility
- Ensure mussel version >= 1.1.0
- Check that patch encoder matches slide encoder requirements
