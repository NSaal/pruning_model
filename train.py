import numpy as np
from model.resnet50 import resnet50,pruing_resnet50
from model.vgg16 import vgg16
import os
import zipfile
import tempfile
import tensorflow as tf
import tensorboard
#tf.enable_eager_execution()
#from models.resnet import resnet50
from tensorflow_model_optimization.sparsity import keras as sparsity
batch_size = 128
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


if tf.keras.backend.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

l = tf.keras.layers

num_train_samples = x_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples /
                   batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))

pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                 final_sparsity=0.90,
                                                 begin_step=2000,
                                                 end_step=end_step,
                                                 frequency=100)
}
def train_pruned_resnet50():
    pruning_model=pruing_resnet50(10)
    pruning_model.summary()
    logdir = ".//logs"
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]
    pruning_model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy'])

    callbacks = [
       sparsity.UpdatePruningStep(),
     sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
    ]

    pruning_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

    score = pruning_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    checkpoint_file = ".//save//pruing_resnet50_withoutopti.h5"
    print('Saving pruned model to: ', checkpoint_file)
    tf.keras.models.save_model(
        pruning_model, checkpoint_file, include_optimizer=False)


def train_resnet50():
    model = resnet50(10)
    model.summary()
    logdir = ".//logs"
    callbacks = [tf.keras.callbacks.TensorBoard(
        log_dir=logdir, profile_batch=0)]


    model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    checkpoint_file = ".//save//resnet50_withoutopti.h5"
    print('Saving pruned model to: ', checkpoint_file)
    tf.keras.models.save_model(
        pruning_model, checkpoint_file, include_optimizer=False)


def train_vgg16():
    model = vgg16(10)
    model.summary()
    logdir = ".//logs"
    callbacks = [tf.keras.callbacks.TensorBoard(
        log_dir=logdir, profile_batch=0)]

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    checkpoint_file = ".//save//resnet50_withoutopti.h5"
    print('Saving pruned model to: ', checkpoint_file)
    tf.keras.models.save_model(
        pruning_model, checkpoint_file, include_optimizer=False)

def main():
    #train_resnet50()
    #train_pruned_resnet50()
    train_vgg16()

if __name__ == "__main__":
    main()
