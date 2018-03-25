import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

def load():
    labels = pd.read_csv("/Users/Sanjana/Desktop/Test/labels.csv")
    y = np.array([1 if l >= 1 else 0 for l in labels['level']])
    X=np.load("/Users/Sanjana/Desktop/Test/X_train.npy")  
    return X, y

def cnn_model(X_train, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size,
                      nb_classes):

    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    stop = EarlyStopping(monitor='val_acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board])

    return model



batch_size = 50
nb_classes = 2
nb_epoch = 30
img_rows, img_cols = 256, 256
channels = 3
nb_filters = 32
kernel_size = (8, 8)
X,y=load()
X = X.reshape(X.shape[0],img_rows,img_cols,channels)
y = y.reshape(y.shape[0],1)
X = X.astype('float32')
y = np_utils.to_categorical(y, nb_classes)
X/=255

model = cnn_model(X, y, kernel_size, nb_filters, channels, nb_epoch, batch_size,
                      nb_classes)
y_pred = model.predict(X)
score = model.evaluate(X, y, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
y_pred = np.argmax(y_pred, axis=1)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)

