from matplotlib import pyplot as plt
import torch
from veinClassifier.complexFilter import complexOutputFilter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def predict_test_images(model, dataset, device):
    model.eval()
    fig, axes = plt.subplots(3, 5, figsize=(18, 7))

    names = ["Prediction", "Filtered"]
    all_metrics = { name:[] for name in names}

    for i in tqdm(range(5)):
        img, segm = dataset[i]
        
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5)
            y_pred = pred.squeeze().cpu().numpy()

        segmentation_pred = y_pred.astype(np.uint)
        segmentation_true = (segm.squeeze() > 0.5).numpy().astype(np.uint)
        segmentation_filtered = complexOutputFilter(segmentation_pred)

        # Flatten for sklearn metrics
        y_true = segmentation_true.flatten()
        y_filtered_flat = segmentation_filtered.flatten()
        y_pred_flat = segmentation_pred.flatten()

        for rowIdx, (y_flat, name) in enumerate(zip([y_pred_flat, y_filtered_flat], names)):
            acc = accuracy_score(y_true, y_flat)
            prec = precision_score(y_true, y_flat, zero_division=0)
            rec = recall_score(y_true, y_flat, zero_division=0)
            f1 = f1_score(y_true, y_flat, zero_division=0)

            all_metrics[name].append({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })

            # Predicted row
            axes[rowIdx, i].imshow(~segmentation_pred.astype(bool), cmap="gray")
            axes[rowIdx, i].set_title(f"{name} {i+1}\n")
            axes[rowIdx, i].set_xlabel(
                f"Acc:{acc:.3f} Prec:{prec:.3f}\n"
                f"Rec:{rec:.3f} F1:  {f1:.3f}"
            )

        # Target row
        axes[2, i].imshow(~segmentation_true.astype(bool), cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title(f"Target {i+1}")

    plt.tight_layout()
    plt.show()

    for name in names:
        # Average metrics over images
        avg_acc = np.mean([m["accuracy"] for m in all_metrics[name]])
        avg_prec = np.mean([m["precision"] for m in all_metrics[name]])
        avg_rec = np.mean([m["recall"] for m in all_metrics[name]])
        avg_f1 = np.mean([m["f1"] for m in all_metrics[name]])

        print(f"\nAverage metrics for {name}:")
        print(f"Accuracy : {avg_acc:.4f}")
        print(f"Precision: {avg_prec:.4f}")
        print(f"Recall   : {avg_rec:.4f}")
        print(f"F1 Score : {avg_f1:.4f}")


