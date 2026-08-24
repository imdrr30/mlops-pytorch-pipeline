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
docker build . -f .\docker\Dockerfile.train -t assignment4train:1.0 
```


To run the Training Image:
```bash
docker run --gpus all --name assignment4train-container -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" assignment4train:1.0
```

The training image writes `checkpoints/model.onnx`. Build and run the lightweight serving image after training:

```bash
docker build . -f .\docker\Dockerfile.serve -t assignment4serve:onnx
docker run --rm --gpus all -v "$(pwd)/checkpoints:/app/checkpoints" --name assignment4serve-container -p 8080:8080 assignment4serve:onnx
```
