import os

import tensorflow as tf

from utils import *


def get_test_dataset():
    """Returns the test dataset.
    Returns:
        tf.data.Dataset: The test dataset.
    """
    ROOT_DIR = "/mnt/disks/persist/RiceLeafs"

    TEST_DIR = os.path.join(ROOT_DIR, "validation")

    IMAGE_SIZE = (224, 224)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        seed=42,
    )

    return test_ds


def test(model_name, tpu_address=None):
    model_dir = "model"

    test_ds = get_test_dataset()

    cluster_resolver, strategy = connect_to_tpu(tpu_address=tpu_address)

    with strategy.scope():
        model = tf.keras.models.load_model(os.path.join(model_dir, model_name))

    model.evaluate(test_ds)


if __name__ == "__main__":
    test("rice_leaf_disease_classifier")
