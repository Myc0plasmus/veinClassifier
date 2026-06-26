from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def eval_model(y_true, y_pred, modelPath):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)

    print("\nClassification report:\n", classification_report(y_true, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))

    return {
            "Model" : modelPath.stem,
            "Accuracy" : acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            }

def eval_model_inline(name, y_true, y_pred, modelPath):

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)

    print(f"[{name}] accuracy={acc:.3f} f1={f1:.3f} precision={prec:.3f} recall={rec:.3f}")

    return {
            "Model" : modelPath.stem,
            "Accuracy" : acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            }
