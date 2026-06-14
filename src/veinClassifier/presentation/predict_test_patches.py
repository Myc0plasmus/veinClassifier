from matplotlib import pyplot as plt
from veinClassifier.featureExtraction import extractImagePatches

def predict_test_patches(model, dataset, patchStatisticFun=None):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        img, segm = dataset[i]

        X = extractImagePatches(img)
        if patchStatisticFun is not None:
            X = patchStatisticFun(X)
        y_pred = model.predict(X)

        H, W = img.shape[1], img.shape[2]
        segmentation_pred = y_pred.reshape(H, W)

        # Predicted row
        axes[0, i].imshow(~(segmentation_pred > 0.5), cmap="gray")
        axes[0, i].axis("off")

        # Actual row
        axes[1, i].imshow(~(segm.squeeze() > 0.5), cmap="gray")
        axes[1, i].axis("off")

    # Optional column titles
    for i in range(5):
        axes[0, i].set_title(f"Prediction {i+1}")
        axes[1, i].set_title(f"Target {i+1}")

    plt.tight_layout()
    plt.show()
