import os
import sys

from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import io

def get_img():
    """
    will return None if no image in clipboard
    """
    try:
        result = subprocess.run(["xclip", "-se", "c", "-t", "image/png", "-o"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        im = Image.open(io.BytesIO(result.stdout))
        im = im.resize((64,64),Image.ANTIALIAS)
        return np.array(im)
    except OSError:
        return None

def load_img(path):
    im = Image.open(path).convert('RGB')
    im = im.resize((64, 64), Image.ANTIALIAS)
    return np.array(im)

def load_imgs(path):
    print("Loading {}...".format(path))
    ret = []
    for file in os.listdir(path):
        # print("\t{}...".format(file), end="")
        # sys.stdout.flush()
        ret.append(load_img(os.path.join(path,file)))
        # print("[OK!]")
    print("Loaded!")
    return np.array(ret)


# load data
cats = load_imgs("res/cat")
hats = load_imgs("res/hat")
plt.imshow(hats[0])
plt.show()