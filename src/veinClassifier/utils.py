from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import numpy as np
import torch
from torch.amp.autocast_mode import autocast

from torch.amp.grad_scaler import GradScaler
from torch import nn
from matplotlib import pyplot as plt


def compute_pos_weight(loader, device):
    total_pos = 0.0
    total_neg = 0.0

    for x_batch, y_batch in loader:
        y_batch = y_batch.to(device)

        total_pos += (y_batch).sum().item()
        total_neg += ((1 - y_batch)).sum().item()

    return max(total_neg / (total_pos + 1e-8), 1.0)

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    all_y = []
    all_pred = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if is_train:
            optimizer.zero_grad()

        with autocast("cuda"):
            logits = model(xb)
            loss = criterion(logits, yb)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5) 
        

        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    all_y = np.concatenate(all_y).ravel()
    all_pred = np.concatenate(all_pred).ravel()
    return float(np.mean(losses)), all_y, all_pred


def train_model(model, train_loader, val_loader, criterion, device, lr=1e-3, epochs=30, weight_decay=0.0):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda")
    
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for ep in range(1, epochs+1):
        tr_loss, tr_y, tr_pred = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        va_loss, va_y, va_pred = run_epoch(model, val_loader, criterion, device, optimizer=None, scaler=None)

        tr_f1 = f1_score(tr_y, tr_pred)
        va_f1 = f1_score(va_y, va_pred)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        # if ep % 5 == 0 or ep == 1:
        print(f"Epoch {ep:02d}: train_loss={tr_loss:.3f} val_loss={va_loss:.3f} train_f1={tr_f1:.3f} val_f1={va_f1:.3f}")

    return model, history

def plot_history(history, title, filename):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title(title + " - loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(f"{filename}_loss.png")
    plt.show()

    plt.figure()
    plt.plot(history["train_f1"], label="train_f1")
    plt.plot(history["val_f1"], label="val_f1")
    plt.title(title + " - F1")
    plt.xlabel("epoch")
    plt.legend()
    plt.savefig(f"{filename}_f1.png")
    plt.show()
