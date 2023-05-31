import os

import tensorflow as tf

from utils import connect_to_tpu


def get_datasets():
    """Returns the training and validation datasets.
    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    ROOT_DIR = "/mnt/disks/persist/RiceLeafs"

    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    VAL_DIR = os.path.join(ROOT_DIR, "validation")

    IMAGE_SIZE = (224, 224)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE)

    print("\n")
    return train_ds, val_ds


def get_augmentation():
    """Returns the augmentation layer."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.Rescaling(1./255)
    ])


def get_model(num_classes=4):
    """Returns the model.
    Args:
        num_classes (int): Number of classes to classify.
    """
    pre_trained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    pre_trained_model.trainable = True

    model = tf.keras.models.Sequential(
        [
            # get_augmentation(),
            pre_trained_model,
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(1000, 3, name="logits"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction"),
        ]
    )

    return model

def get_callbacks():
    """Returns the callbacks for the model.
    Returns:
        list: A list containing the callbacks for the model.
    """
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(os.getcwd(), "logs"),
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * tf.math.exp(-0.1) if epoch > 20 else lr
        )
    ]

    return callbacks

def train(
    tpu_address: str = None,
):
    tf.keras.backend.clear_session()
    print("\n")

    cluster_resolver, strategy = connect_to_tpu(tpu_address=tpu_address)

    print("Preparing Datasets...\n")

    train_ds, val_ds = get_datasets()

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print("Augmenting Datasets...\n")
    data_augmentation = get_augmentation()
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y))

    batch_size = 200

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().repeat().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().repeat().prefetch(buffer_size=AUTOTUNE)
    

    with strategy.scope():
        model = get_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

    steps_per_epoch = 40000 // batch_size
    validation_steps = 10000 // batch_size

    print("Fitting Model...\n")

    with strategy.scope():
        history = model.fit(
            train_ds,
            epochs=200,
            batch_size=batch_size,
            validation_data=val_ds,
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            callbacks=get_callbacks(),
        )

    model.save(os.path.join(os.getcwd(), "model", "rice_leaf_disease_classifier"))


if __name__ == "__main__":
    train()
