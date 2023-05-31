import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from train import *
from test import *

if __name__ == "__main__":
    # train(tpu_address="local")
    test(model_name="rice_leaf_disease_classifier")
