from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    all_y = []
    all_pred = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = (torch.sigmoid(logits) > 0.5).float()

        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(preds.detach().cpu().numpy())

    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    return float(np.mean(losses)), all_y, all_pred


def train_model(model, train_loader, val_loader, lr=1e-3, epochs=30, weight_decay=0.0):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for ep in range(1, epochs+1):
        tr_loss, tr_y, tr_pred = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_y, va_pred = run_epoch(model, val_loader, criterion, optimizer=None)

        tr_f1 = f1_score(tr_y.flatten(), tr_pred.flatten())
        va_f1 = f1_score(va_y.flatten(), va_pred.flatten())

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        if ep % 5 == 0 or ep == 1:
            print(f"Epoch {ep:02d}: train_loss={tr_loss:.3f} val_loss={va_loss:.3f} train_f1={tr_f1:.3f} val_f1={va_f1:.3f}")

    return model, history
