from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.architectures import build_model
from ml.data.dataset import SentimentDataset
from ml.data.text_processing import BertTokenizerEncoder
from ml.evaluation.metrics import compute_metrics
from ml.training.checkpoint import (
    load_model_from_checkpoint,
    next_run_dir,
    save_checkpoint,
)
from ml.training.optimizer import build_optimizer, select_device
from ml.constants import CLASS_NAMES


SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
IMAGE_SIZE = 128
DEVICE = select_device("cpu")


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_splits():
    df = pd.read_csv("data/processed/mvsa_single/samples.csv")

    if "split" not in df.columns:
        raise ValueError("Dataset CSV must contain a split column.")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Training or validation split is empty.")

    return train_df, val_df


def extract_features(model, loader, device, description):
    model.image_encoder.eval()
    model.text_encoder.eval()

    image_features = []
    text_features = []
    labels = []

    start = time.time()

    with torch.inference_mode():
        for index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            image_feat = model.image_encoder(images)
            text_feat = model.text_encoder(input_ids, attention_mask)

            image_features.append(image_feat.cpu())
            text_features.append(text_feat.cpu())
            labels.append(batch["label"].cpu())

            if index % 10 == 0 or index == len(loader):
                print(
                    f"{description}: {index}/{len(loader)} batches completed"
                )

    print(
        f"{description} feature extraction: "
        f"{time.time() - start:.1f} seconds"
    )

    return (
        torch.cat(image_features),
        torch.cat(text_features),
        torch.cat(labels).long(),
    )


def evaluate_fusion(model, features, device):
    image_feat, text_feat, labels = features

    model.fusion.eval()

    with torch.inference_mode():
        image_feat = image_feat.to(device)
        text_feat = text_feat.to(device)

        logits = model.fusion(image_feat, text_feat)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return compute_metrics(
        labels.numpy(),
        predictions.cpu().numpy(),
        probabilities.cpu().numpy(),
        class_names=CLASS_NAMES,
    )


def main():
    seed_everything(SEED)
    print("SENTRA AI - CNN + BERT FUSION TRAINING")
    print("Device:", DEVICE)

    print("\n[1/7] Loading trained encoders...")

    cnn, cnn_ckpt = load_model_from_checkpoint(
        "artifacts/runs/cnn_mvsa_single_v7/best.pt",
        DEVICE,
    )

    bert, bert_ckpt = load_model_from_checkpoint(
        "artifacts/runs/bert_cached_mvsa_single_v1/best.pt",
        DEVICE,
    )

    print("ResNet18 checkpoint loaded.")
    print("BERT checkpoint loaded.")

    print("\n[2/7] Building multimodal architecture...")

    model = build_model(
        "cnn_bert",
        image_backbone="resnet18",
        image_channels=[32, 64, 128, 256],
        image_pretrained=False,
        pretrained_name=None,
        bert_config=bert_ckpt["model_config"]["bert_config"],
        freeze_encoder=True,
        proj_dim=256,
        fusion_hidden=[256, 128],
        dropout=0.3,
        use_interaction=False,
    ).to(DEVICE)

    cnn_image_state = {
        key.removeprefix("image_encoder."): value
        for key, value in cnn_ckpt["state_dict"].items()
        if key.startswith("image_encoder.")
    }

    image_result = model.image_encoder.load_state_dict(
        cnn_image_state,
        strict=True,
    )

    bert_state = {
        key.removeprefix("text_encoder.bert."): value
        for key, value in bert_ckpt["state_dict"].items()
        if key.startswith("text_encoder.bert.")
    }

    bert_result = model.text_encoder.bert.load_state_dict(
        bert_state,
        strict=True,
    )

    for parameter in model.image_encoder.parameters():
        parameter.requires_grad = False

    for parameter in model.text_encoder.parameters():
        parameter.requires_grad = False

    model.image_encoder.eval()
    model.text_encoder.eval()

    print("Image transfer:", image_result)
    print("Text transfer:", bert_result)
    print("Total parameters:", model.count_parameters())
    print(
        "Trainable parameters:",
        model.count_parameters(trainable_only=True),
    )

    print("\n[3/7] Loading dataset and tokenizer...")

    train_df, val_df = load_splits()

    config = yaml_config = None
    import yaml

    with open("configs/default.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    encoder = BertTokenizerEncoder(
        config["data"]["bert_tokenizer"],
        max_len=config["data"]["max_text_len"],
    )

    train_dataset = SentimentDataset(
        train_df,
        mode="multimodal",
        text_encoder=encoder,
        image_size=IMAGE_SIZE,
        train=False,
    )

    val_dataset = SentimentDataset(
        val_df,
        mode="multimodal",
        text_encoder=encoder,
        image_size=IMAGE_SIZE,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    print("\n[4/7] Extracting and caching frozen encoder features...")

    train_features = extract_features(
        model, train_loader, DEVICE, "TRAIN"
    )

    val_features = extract_features(
        model, val_loader, DEVICE, "VALIDATION"
    )

    train_tensor_dataset = TensorDataset(*train_features)

    feature_loader = DataLoader(
        train_tensor_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    print("\nCached feature dimensions:")
    print("Image:", tuple(train_features[0].shape))
    print("Text:", tuple(train_features[1].shape))

    print("\n[5/7] Preparing fusion optimizer and loss...")

    optimizer = build_optimizer(
        model.fusion,
        name="adamw",
        lr=0.0003,
        weight_decay=0.01,
    )

    class_counts = torch.bincount(
        train_features[2],
        minlength=len(CLASS_NAMES),
    ).float()

    class_weights = (
        class_counts.sum() /
        (len(CLASS_NAMES) * class_counts.clamp(min=1))
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    run_dir = next_run_dir(
        "artifacts/runs",
        "cnn_bert",
        "mvsa_single",
    )

    print("Experiment folder:", run_dir)

    with open(run_dir / "architecture.json", "w") as file:
        json.dump(model.config, file, indent=2)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    extra = {
        "text_encoder": encoder.state(),
        "image_size": IMAGE_SIZE,
        "dataset": "mvsa_single",
        "fusion_training": True,
        "frozen_image_encoder": True,
        "frozen_text_encoder": True,
        "source_cnn_checkpoint": "cnn_mvsa_single_v7",
        "source_bert_checkpoint": "bert_cached_mvsa_single_v1",
    }

    print("\n[6/7] Training fusion network...")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.fusion.train()

        total_loss = 0.0
        total_samples = 0

        for image_feat, text_feat, labels in feature_loader:
            image_feat = image_feat.to(DEVICE)
            text_feat = text_feat.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model.fusion(image_feat, text_feat)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.fusion.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        val_metrics = evaluate_fusion(
            model,
            val_features,
            DEVICE,
        )

        val_f1 = val_metrics["f1_macro"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Macro F1: {val_f1:.4f} | "
            f"Time: {time.time() - epoch_start:.1f}s"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                extra,
            )

            print("  BEST CHECKPOINT SAVED")

        else:
            patience_counter += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics,
            extra,
        )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\n[7/7] Saving experiment report...")

    with open(run_dir / "history.json", "w") as file:
        json.dump(history, file, indent=2)

    summary = {
        "model": "cnn_bert",
        "dataset": "mvsa_single",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "total_parameters": model.count_parameters(),
        "trainable_parameters": model.count_parameters(
            trainable_only=True
        ),
        "run_directory": str(run_dir),
    }

    with open(run_dir / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)

    print("\nFUSION TRAINING COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
