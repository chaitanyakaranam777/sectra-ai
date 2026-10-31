"""Train BERT using cached frozen-encoder embeddings."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.architectures import build_model
from ml.data.dataset import SentimentDataset, make_loader
from ml.data.text_processing import BertTokenizerEncoder
from ml.evaluation.metrics import compute_metrics
from ml.training.callbacks import write_json_atomic
from ml.training.checkpoint import next_run_dir, save_checkpoint
from ml.training.losses import build_loss, compute_class_weights
from ml.training.run import load_config
from ml.training.trainer import set_seed


@torch.no_grad()
def cache_embeddings(model, dataset, batch_size, device, name):
    """Run the frozen BERT encoder once and retain CLS embeddings in RAM."""
    model.text_encoder.eval()
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False)
    embeddings, labels = [], []

    print(f"\nCaching {name} embeddings ({len(dataset)} samples)...")
    start = time.time()

    for step, batch in enumerate(loader, 1):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)

        features = model.text_encoder(ids, mask)
        embeddings.append(features.cpu())
        labels.append(batch["label"].cpu())

        if step % 25 == 0 or step == len(loader):
            print(f"{name}: {step}/{len(loader)} batches")

    result = (torch.cat(embeddings), torch.cat(labels))
    print(f"{name} cache completed in {(time.time() - start) / 60:.1f} minutes.")
    return result


def evaluate_head(model, embeddings, labels, loss_fn, batch_size, device):
    model.classifier.eval()
    loader = DataLoader(
        TensorDataset(embeddings, labels),
        batch_size=batch_size,
        shuffle=False
    )

    all_probs, all_labels = [], []
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model.classifier(features)
            total_loss += loss_fn(logits, targets).item() * len(targets)
            count += len(targets)

            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(targets.cpu())

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = probs.argmax(axis=1)

    metrics = compute_metrics(y_true, y_pred, probs)
    metrics["loss"] = total_loss / count
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--dataset", default="mvsa_single")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs", default="artifacts/runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device(args.device)
    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    df = pd.read_csv(args.samples)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Non-empty train and validation splits are required.")

    encoder = BertTokenizerEncoder(
        data_cfg["bert_tokenizer"],
        data_cfg["max_text_len"]
    )

    model_params = dict(cfg["models"]["bert"])
    model_params["freeze_encoder"] = True

    model = build_model("bert", **model_params).to(device)
    model.text_encoder.eval()

    for parameter in model.text_encoder.parameters():
        parameter.requires_grad = False

    train_ds = SentimentDataset(
        train_df, "text", encoder,
        int(data_cfg["image_size"]), train=False
    )
    val_ds = SentimentDataset(
        val_df, "text", encoder,
        int(data_cfg["image_size"]), train=False
    )

    # BERT runs only once for each sample in each split.
    train_x, train_y = cache_embeddings(
        model, train_ds, args.encoder_batch_size, device, "TRAIN"
    )
    val_x, val_y = cache_embeddings(
        model, val_ds, args.encoder_batch_size, device, "VALIDATION"
    )

    weights = compute_class_weights(train_ds.labels)
    loss_fn = build_loss(weights, 0.0, device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    run_dir = next_run_dir(args.runs, "bert_cached", args.dataset)
    run_dir.mkdir(parents=True, exist_ok=True)

    extra = {
        "text_encoder": encoder.state(),
        "image_size": int(data_cfg["image_size"]),
        "dataset": args.dataset
    }

    write_json_atomic(run_dir / "config.json", {
        "model": "bert",
        "training_mode": "cached_frozen_encoder",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_batch_size": args.encoder_batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "model_params": model.config
    })
    write_json_atomic(run_dir / "architecture.json", model.summary())

    best_f1 = -1.0
    best_epoch = 0
    patience = 5
    stale_epochs = 0
    history = []

    print(f"\nStarting cached BERT head training.")
    print(f"Run directory: {run_dir}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.classifier.train()

        total_loss, correct, count = 0.0, 0, 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(features)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.classifier.parameters(), 1.0
            )
            optimizer.step()

            total_loss += loss.item() * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)

        train_loss = total_loss / count
        train_acc = correct / count

        val_metrics = evaluate_head(
            model, val_x, val_y, loss_fn,
            args.batch_size, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["f1_macro"],
            "duration_s": time.time() - start
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_acc:.4f} | "
            f"Val acc {val_metrics['accuracy']:.4f} | "
            f"Val macro F1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1 + 0.0001:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            stale_epochs = 0

            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                record | {"monitor": "val_macro_f1"},
                extra
            )
        else:
            stale_epochs += 1

        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            record,
            extra
        )

        write_json_atomic(run_dir / "history.json", {
            "best_epoch": best_epoch,
            "epochs": history
        })

        write_json_atomic(run_dir / "status.json", {
            "state": "running",
            "epoch": epoch,
            "total_epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_metric": best_f1
        })

        if stale_epochs >= patience:
            print("Early stopping: validation macro F1 did not improve.")
            break

    summary = {
        "model": "bert",
        "training_mode": "cached_frozen_encoder",
        "device": str(device),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "parameters_total": model.count_parameters(),
        "parameters_trainable": model.count_parameters(trainable_only=True)
    }

    write_json_atomic(run_dir / "summary.json", summary)
    write_json_atomic(run_dir / "status.json", {
        "state": "completed",
        **summary
    })

    print("\nTraining completed.")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved run: {run_dir}")


if __name__ == "__main__":
    main()
