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
docker build . -f .\docker\Dockerfile.train -t mlops-train:1.0 
```


To run the Training Image:
```bash
docker run --gpus all --name mlops-train-container -v "$(pwd)/data:/app/data" -v "$(pwd)/checkpoints:/app/checkpoints" mlops-train:1.0
```

The training image writes `checkpoints/model.onnx`. Build and run the lightweight serving image after training:

```bash
docker build . -f .\docker\Dockerfile.serve -t mlops-serve:onnx
docker run --rm --gpus all -v "$(pwd)/checkpoints:/app/checkpoints" --name mlops-serve-container -p 8080:8080 mlops-serve:onnx
```

Kubernetes on Docker Desktop
----------------------------

The Kubernetes training Job uses host-backed volumes so the existing repository directories are mounted in the same way as the Docker commands above. Apply the storage resources before starting the Job:

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/training-job.yaml
```

The host paths in `k8s/storage.yaml` target this repository at `D:\Projects\mlops-pytorch-pipeline`. Update those paths if the repository is located elsewhere. This local configuration runs the training Job on CPU.

To start the model serving layer in Kubernetes after training:

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl get svc -n ml-training
```

Then expose it locally:

```powershell
kubectl port-forward svc/serving-service -n ml-training 8080:80
```

Serve the model on Kubernetes and test it locally:

```powershell
curl.exe -X POST http://localhost:8080/predict -F "image=@tests/airplane3.png"
```
