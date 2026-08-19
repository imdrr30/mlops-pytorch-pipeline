# mlops-pytorch-pipeline

Minimal MLOps scaffold for a PyTorch training and serving pipeline.

Structure:

- `src/` - training, model, dataset and serving entrypoints
- `configs/` - config files
- `docker/` - Dockerfiles for training and serving
- `k8s/` - Kubernetes manifests for training and serving
- `requirements/` - dependency lists for images
- `tests/` - simple unit tests

Quick start
-----------

Build the training image:

```bash
docker build -f docker/Dockerfile.train -t mlops-train:latest .
```
