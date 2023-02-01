"""
CS/DS 549 Spring 2023 Programming and Model Training Assignment

The goal is to define a better model and training hyperparameters to beat the minimum
required evaluation/validation accuracy of 0.82  at the very least, but also to compete
in the class challenge for best training results.

Only edit code between the comments:
#########################
# Edit code here
# vvvvvvvvvvvvvvvvvvvvvvv
<code>
# ^^^^^^^^^^^^^^^^^^^^^^^
"""
import wandb
from wandb.keras import WandbMetricsLogger

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt



if __name__ == '__main__':

    # Leave entity="bu-spark-ml" and project="hw1_spring2023"
    # put your BU username in the `group=` parameter
    wandb.init(
        project="hw1_spring2023",  # Leave this as 'hw1_spring2023'
        entity="bu-spark-ml",  # Leave this
        group="hjc5283",  # <<<<<<< Put your BU username here
        notes="test-1-4"  # <<<<<<< You can put a short note here
    )

    """
    Use tfds to load the CIFAR10 dataset and visualize the images and train.

    The datasets used are:
    https://www.tensorflow.org/datasets/catalog/cifar10
    https://www.tensorflow.org/datasets/catalog/cifar10_corrupted

    tfds.load() whill first check if the dataset is already downloaded to the
    path in `data_dir`. If not, it will download the dataset to that path..
    """
    # Load the CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    (ds_cifar10_train, ds_cifar10_test), ds_cifar10_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        data_dir='/projectnb/ds549/datasets/tensorflow_datasets',
        shuffle_files=True, # load in random order
        as_supervised=True, # Include labels
        with_info=True, # Include info
    )

    # Optionally uncomment the next 3 lines to visualize random samples from each dataset
    #fig_train = tfds.show_examples(ds_cifar10_train, ds_cifar10_info)
    #fig_test = tfds.show_examples(ds_cifar10_test, ds_cifar10_info)
    #plt.show()  # Display the plots

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    # Prepare cifar10 training dataset
    ds_cifar10_train = ds_cifar10_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_cifar10_train = ds_cifar10_train.cache()     # Cache data
    ds_cifar10_train = ds_cifar10_train.shuffle(ds_cifar10_info.splits['train'].num_examples)
    ds_cifar10_train = ds_cifar10_train.batch(256)  # <<<<< To change batch size, you have to change it here
    ds_cifar10_train = ds_cifar10_train.prefetch(tf.data.AUTOTUNE)

    # Prepare cifar10 test dataset
    ds_cifar10_test = ds_cifar10_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_cifar10_test = ds_cifar10_test.batch(256)    # <<<<< To change batch size, you have to change it here
    ds_cifar10_test = ds_cifar10_test.cache()
    ds_cifar10_test = ds_cifar10_test.prefetch(tf.data.AUTOTUNE)

    # Define the model here
    model = tf.keras.models.Sequential([
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(1024, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        #####################################
        # Edit code here -- Update the model definition
        # You will need a dense last layer with 10 output channels to classify the 10 classes
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        layers.Flatten(),
        layers.Dense(512, activation=layers.LeakyReLU(alpha=0.05), kernel_regularizer=regularizers.l2(0.006)),
        layers.Dense(128, activation=layers.LeakyReLU(alpha=0.05), kernel_regularizer=regularizers.l2(0.006)),
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    # Log the training hyper-parameters for WandB
    # If you change these in model.compile() or model.fit(), be sure to update them here.
    wandb.config = {
        #####################################
        # Edit these as desired
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 28,
        "batch_size": 256
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_cifar10_train,
        epochs=28,
        validation_data=ds_cifar10_test,
        callbacks=[WandbMetricsLogger()]
    )

    wandb.finish()
