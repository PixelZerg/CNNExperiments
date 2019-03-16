import os
import sys

from PIL import Image
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Conv3D, Conv2D, Dense, Flatten

import io

def get_img():
    """
    will return None if no image in clipboard
    """
    try:
        result = subprocess.run(["xclip", "-se", "c", "-t", "image/png", "-o"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        im = Image.open(io.BytesIO(result.stdout)).convert('RGB')
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


# load data (X)
cats = load_imgs("res/cat")
hats = load_imgs("res/hat")

# outs (Y)
cat_outs = np.rot90(np.vstack((np.ones(len(cats)),np.zeros(len(cats))))) # [1, 0]s
hat_outs = np.rot90(np.vstack((np.zeros(len(hats)),np.ones(len(hats))))) # [0, 1]s

# concatenate images to form training data
X = np.concatenate((cats,hats))
Y = np.concatenate((cat_outs, hat_outs))

# shuffle training data
shuff = np.arange(len(X))
np.random.shuffle(shuff)

X = X[shuff]
Y = Y[shuff]

# i=0
# while True:
#     # plt.imshow(X[np.random.choice(len(X))])
#     print(Y[i])
#     plt.imshow(X[i])
#     plt.show()
#     i+=1

# model
model = Sequential()
model.add(Conv2D(64,kernel_size=32,activation="relu",input_shape=(64,64,3)))
model.add(Conv2D(32,kernel_size=8,activation="relu"))
model.add(Conv2D(16,kernel_size=3,activation="relu", input_shape=(32,32,1)))
model.add(Flatten())
model.add(Dense(2,activation="softmax"))

# compile and train model
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(X,Y,epochs=1000)

# test model
while True:
    input("Test model!")
    print(model.predict(np.array([get_img()])))
