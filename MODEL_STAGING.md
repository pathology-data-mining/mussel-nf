# Model Staging for Containerization

## Current Approach
Models are referenced by absolute paths in `nextflow.config`:
```groovy
featurize {
    model_paths {
        ctranspath = "/path/to/ctranspath.pth"
        ...
    }
}
```

## For Containerization

### Option 1: Volume Mounts (Recommended)
Mount model directories in your container:

```bash
# Docker
docker run -v /path/to/models:/models ...

# Apptainer
apptainer run --bind /path/to/models:/models ...
```

Update config for the container profile:
```groovy
profiles {
    docker {
        params {
            featurize.model_paths {
                ctranspath = "/models/ctranspath.pth"
                optimus = "/models/optimus.pkl"
            }
        }
    }
}
```

### Option 2: Bake Models into Container
Include models in your Dockerfile:
```dockerfile
FROM mskmind/mussel:current
COPY models/ /opt/models/
ENV MODEL_DIR=/opt/models
```

### Option 3: Download at Runtime
Add a setup process:
```nextflow
process DOWNLOAD_MODELS {
    storeDir "${params.model_cache_dir}"
    
    output:
    path "*.pth", emit: models
    
    script:
    """
    wget https://example.com/models/ctranspath.pth
    """
}
```

## Why Not Stage in Work Directory?

Nextflow's work directory is designed for:
- Input data that changes per task
- Temporary outputs

Models are:
- Large (100MB+)
- Shared across all tasks  
- Don't change between runs

Staging them would:
- Copy models for every task (wasteful)
- Slow down pipeline
- Fill up disk space

## Best Practice
Use absolute paths + container volume mounts. This is the standard Nextflow pattern for large reference files.
