import numpy as np
from matplotlib import pyplot as plt


def plot_confusion_matrix(df_confusion, classes, title='Confusion matrix', cmap="seismic"):
    fig, ax = plt.subplots()
    cax = ax.matshow(df_confusion, cmap=cmap) # imshow
    for (i, j), z in np.ndenumerate(df_confusion):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #ax.title(title)
    fig.colorbar(cax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, rotation=45)
    ax.set_xticklabels(classes)

    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    #ax.tight_layout()
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig, ax