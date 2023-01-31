import glob
import os.path
import shutil

import numpy as np

imglist = glob.glob("animal3-results-backup/results-0.001/**/contr*.png", recursive=True)
# os.makedirs("contr_plots")
for i in imglist:
    num = os.path.basename(os.path.dirname(i))
    shutil.copy(i, "contr_plots/" + num + "-cont_plot.png")
