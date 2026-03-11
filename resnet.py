import argparse
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser(description="ResNet-18 classification for pneumonia")
    parser.add_argument("--train_dir", required=True, help="Path to one experiment train folder")
    parser.add_argument("--test_dir", required=True, help="Path to chest_xray/test")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exp_name = Path(args.train_dir).name
    print(f"=== Start ResNet-18 pipeline: {exp_name} ===")

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(args.test_dir, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Model
    # - weights=None for scratch
    # - weights=ResNet18_Weights.IMAGENET1K_V1 for pretrained
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train and eval
    best_acc = 0.0
    best_metrics = {}
    best_probs = []
    best_labels = []

    epoch_metrics_history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                test_loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} [Test]",
                leave=False,
            ):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        sensitivity = recall_score(all_labels, all_preds, pos_label=1)
        f1 = f1_score(all_labels, all_preds, pos_label=1)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        auc_score = roc_auc_score(all_labels, all_probs)

        epoch_metrics_history.append({
            "Epoch": epoch + 1,
            "Train_Loss": epoch_loss,
            "Test_Accuracy": acc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "F1_Score": f1,
            "AUC": auc_score,
        })

        print(
            f"Epoch {epoch + 1} -> "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"Sens: {sensitivity:.4f} | "
            f"Spec: {specificity:.4f} | "
            f"F1: {f1:.4f} | "
            f"AUC: {auc_score:.4f}"
        )

        # Save best epoch by accuracy
        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                "Epoch": epoch + 1,
                "Accuracy": acc,
                "Sensitivity": sensitivity,
                "Specificity": specificity,
                "F1-Score": f1,
                "AUC": auc_score,
            }
            best_probs = all_probs
            best_labels = all_labels

    # Save outputs
    print("\n" + "=" * 50)
    print(f"Training done. Best epoch: {best_metrics['Epoch']}")
    print(f"Accuracy:    {best_metrics['Accuracy'] * 100:.2f}%")
    print(f"Sensitivity: {best_metrics['Sensitivity'] * 100:.2f}%")
    print(f"Specificity: {best_metrics['Specificity'] * 100:.2f}%")
    print(f"F1-Score:    {best_metrics['F1-Score'] * 100:.2f}%")
    print(f"AUC:         {best_metrics['AUC']:.4f}")
    print("=" * 50 + "\n")

    metrics_filename = f"Metrics_{exp_name}.csv"
    with open(metrics_filename, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Epoch",
                "Train_Loss",
                "Test_Accuracy",
                "Sensitivity",
                "Specificity",
                "F1_Score",
                "AUC",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_metrics_history)
    print(f"Saved training metrics to: {metrics_filename}")

    roc_filename = f"ROC_Data_{exp_name}.csv"
    with open(roc_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True_Label", "Pred_Prob"])
        for label, prob in zip(best_labels, best_probs):
            writer.writerow([label, prob])
    print(f"Saved ROC data to: {roc_filename}")


if __name__ == "__main__":
    main()