import os

import matplotlib.pyplot as plt

if "DOCUTILSCONFIG" in os.environ:
    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
    figsize = 10.0
else:
    get_ipython().run_line_magic("matplotlib", "notebook")  # noqa: F821
    figsize = 6.0
plt.rcParams["figure.figsize"] = [figsize, figsize]
