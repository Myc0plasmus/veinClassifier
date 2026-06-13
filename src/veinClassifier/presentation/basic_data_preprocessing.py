import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Name | Surname | index
    -|-|-
    Natan | Jabłoński | 155621
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data import
    """)
    return


@app.cell
def _():
    from matplotlib import pyplot as plt
    from src.dataset import EyeDataset

    dataset = EyeDataset("../../data/healthy/", "../../data/healthy_manualsegm")

    img, segm = dataset[0]
    return img, plt, segm


@app.cell
def _():
    import os
    print(os.getcwd())
    return


@app.cell
def _():
    import sys
    print(sys.path[:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Eye fundus image
    """)
    return


@app.cell
def _(img):
    from matplotlib import pyplot as plt
    plt.imshow(img.permute(1,2,0))
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Manual segmentation image
    """)
    return


@app.cell
def _(plt, segm):
    plt.imshow(~(segm.squeeze() > 0.5), cmap="gray")
    plt.show() #AUC ROC
    return


if __name__ == "__main__":
    app.run()
