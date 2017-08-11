import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# from keras import backend as K
#
# K.set_image_dim_ordering('th')

retrain = 0


def show_images_in_table(images, table_size, fig_size=(10, 10), cmap=None, titles=None):
    sizex = table_size[0]
    sizey = table_size[1]
    fig, imtable = plt.subplots(sizey, sizex, figsize=fig_size, squeeze=False)
    for j in range(sizey):
        for i in range(sizex):
            im_idx = i + j * sizex
            if isinstance(cmap, (list, tuple)):
                imtable[j][i].imshow(images[im_idx], cmap=cmap[i])
            else:
                im = images[im_idx]
                if len(im.shape) == 3:
                    imtable[j][i].imshow(im)
                else:
                    imtable[j][i].imshow(im, cmap='gray')
            imtable[j][i].axis('off')
            if titles is not None:
                imtable[j][i].set_title(titles[im_idx], fontsize=20)

    plt.show()


# define baseline model
def train_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


def preview_pred_images(len_test_set):
    prob = []
    for i in range(len_test_set, len_test_set + 12):
        prob.append(model.predict(X_test[i].reshape(1, 28, 28, 1)))

    # print prob
    predictions = []
    for j in prob:
        # print j
        prediction = int()
        for i in range(10):
            # print j[i][0]
            if int(round(j[0][i])) == 1:
                prediction = i
        predictions.append(prediction)

    test_set = [X_test[i].reshape(28, 28) for i in range(len_test_set, len_test_set + 12)]
    # print predictions
    show_images_in_table(test_set, (4, 3), (10, 10), "gray", predictions)


fname = "mnist_baseline.h5"

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

size = len(train)
slice_percentage = floor(size * 0.8) - 1

X_train = train.ix[:, 1:].values.astype('float32')
y_train = train.ix[:, 0].values.astype('int32')
#
# x_test = train.ix[slice_percentage + 1:, 1:].values.astype('float32')
# y_test = train.ix[slice_percentage + 1:, 0].values.astype('int32')


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# print y_train


# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
# x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
num_classes = 10

num_pixels = 784
if not os.path.exists(fname) or retrain:
    print "Model does not exist...Training..."
    model = train_model()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=200, verbose=1)
    # Final evaluation of the model
    model.save(fname)
else:
    model = load_model(fname)

# scores = model.evaluate(X_val, y_val, verbose=2)
# print("Error: %.2f%%" % (100 - scores[1] * 100))
# print("Accuracy: %.2f%%" % (scores[1] * 100))

X_test = test.values.astype('float32')
# print X_test.shape
# print X_test.shapen
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test /= 255

# preview_pred_images(12)
prob = []
for i in range(len(X_test)):
    prob.append(model.predict(X_test[i].reshape(1, 28, 28, 1)))

with open('results.csv', "w") as f:
    f.write("ImageId,Label\n")
    predictions = []
    id = 1
    for j in prob:
        # print j
        prediction = int()
        for i in range(10):
            # print j[i][0]
            if int(round(j[0][i])) == 1:
                prediction = i
        predictions.append(prediction)
        f.write(str(id) + "," + str(prediction) + "\n")
        id += 1
    print len(predictions)
