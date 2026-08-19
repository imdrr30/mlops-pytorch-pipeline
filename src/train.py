import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config_path = args.config or os.environ.get("TRAINING_CONFIG") or os.environ.get("CONFIG_PATH")
    if not config_path:
        default_path = Path("/app/configs/training_config.yaml")
        config_path = default_path if default_path.exists() else Path("configs/training_config.yaml")

    config = load_config(str(config_path))
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    output_config = config["output"]
    device_name = training_config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")

    model = get_model(
        architecture=model_config["architecture"],
        num_classes=model_config["num_classes"],
    ).to(device)
    train_loader, val_loader = get_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=training_config["batch_size"],
        num_workers=training_config.get("num_workers", 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(output_config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / output_config["model_name"]
    metrics_path = Path(output_config.get("metrics_path", checkpoint_dir / "metrics.jsonl"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        for epoch in range(training_config["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            log_entry = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
            }
            metrics_file.write(json.dumps(log_entry) + "\n")
            metrics_file.flush()
            print(json.dumps(log_entry), flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "model_config": model_config,
                }, checkpoint_path)
                print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint_path)}), flush=True)
            else:
                patience_counter += 1
                if patience_counter >= training_config["early_stopping_patience"]:
                    print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                    break
    print(json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}), flush=True)


if __name__ == "__main__":
    main()
