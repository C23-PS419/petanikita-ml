import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


class LeafDiseaseClassifier(tf.keras.Model):
    """A basic transfer learned model for classifying leaf diseases."""

    def __init__(
        self,
        num_classes,
        model="MobileNetV2",
        weights="imagenet",
        name="LeafDiseaseClassifier",
    ):
        super(LeafDiseaseClassifier, self).__init__(name=name)

        self.num_classes = num_classes

        self.pre_trained_model, self.preprocess_input = self._get_pre_trained_model(
            model, weights
        )
        self.pre_trained_model.trainable = False

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.fc1 = tf.keras.layers.Dense(1000, name="fc1")
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.fc2 = tf.keras.layers.Dense(500, name="fc2")
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.prediction = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="prediction"
        )

    def call(self, inputs):
        if self.pre_trained_model is None:
            raise NotImplementedError("You must implement this method.")

        x = self.preprocess_input(inputs)
        x = self.pre_trained_model(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return self.prediction(x)

    def _get_pre_trained_model(self, model="MobileNetV2", weights="imagenet"):
        supported_models = ["MobileNetV2", "VGG16"]
        if model not in supported_models:
            raise ValueError(
                f"Unsupported model {model}, supported models are {supported_models}"
            )

        if model == "MobileNetV2":
            return (
                tf.keras.applications.MobileNetV2(
                    input_shape=(224, 224, 3), include_top=False, weights=weights
                ),
                tf.keras.applications.mobilenet_v2.preprocess_input,
            )
        elif model == "VGG16":
            return (
                tf.keras.applications.VGG16(
                    input_shape=(224, 224, 3), include_top=False, weights=weights
                ),
                tf.keras.applications.vgg16.preprocess_input,
            )


class CustomConvolutionLayer(tf.keras.layers.Layer):
    """Custom Convolution Layer for debugging"""

    def __init__(self, units, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(units, kernel_size, activation="relu")
        self.pool = tf.keras.layers.MaxPool2D()

    def call(self, inputs):
        x = self.conv(inputs)
        return self.pool(x)


class CustomModel(tf.keras.Model):
    """Custom Model for debugging"""

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.preprocess_input = (tf.keras.layers.Rescaling(1.0 / 255),)
        self.conv1 = CustomConvolutionLayer(32, 3)
        self.conv2 = CustomConvolutionLayer(64, 3)
        self.conv3 = CustomConvolutionLayer(128, 3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.prediction = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.preprocess_input(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.prediction(x)
