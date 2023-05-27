import os

import tensorflow as tf

from model import LeafDiseaseClassifier


def get_datasets():
    """Returns the training and validation datasets.
    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    ROOT_DIR = "gs://askdhgakwhd1/RiceLeafs"

    TRAIN_DIR = os.path.join(ROOT_DIR, "train")
    VAL_DIR = os.path.join(ROOT_DIR, "validation")

    IMAGE_SIZE = (224, 224)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(VAL_DIR, image_size=IMAGE_SIZE)

    print("\n")
    return train_ds, val_ds


def get_model(num_classes=4):
    """Returns the model.
    Args:
        num_classes (int): Number of classes to classify.
    """
    pre_trained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
    )

    for layer in pre_trained_model.layers:
        layer.trainable = True

    model = tf.keras.models.Sequential(
        [
            pre_trained_model,
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(1000, 3, name="logits"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction"),
        ]
    )

    # model = LeafDiseaseClassifier(num_classes=3, input_shape=(512, 512, 3))

    return model


def connect_to_tpu(tpu_address: str = None):
    if tpu_address is not None:
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=tpu_address
        )
        if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        print("Running on TPU ", cluster_resolver.master())
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        return cluster_resolver, strategy
    else:
        try:
            cluster_resolver = (
                tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            )
            strategy = tf.distribute.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            return cluster_resolver, strategy
        except:
            print("WARNING: No TPU detected.")
            mirrored_strategy = tf.distribute.MirroredStrategy()
            return None, mirrored_strategy


if __name__ == "__main__":
    print("\n")

    cluster_resolver, strategy = connect_to_tpu("node-2")

    print("Preparing Datasets...\n")

    train_ds, val_ds = get_datasets()

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    train_ds = strategy.experimental_distribute_dataset(train_ds)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    with strategy.scope():
        model = get_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", "sparse_categorical_accuracy"],
        )

    batch_size = 200

    steps_per_epoch = 60000 // batch_size
    validation_steps = 10000 // batch_size

    print("Fitting Model...\n")

    history = model.fit(
        train_ds,
        epochs=30,
        batch_size=batch_size,
        validation_data=val_ds,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
    )

    model.save(os.path.join(os.getcwd(), "model", "rice_leaf_disease_classifier"))
