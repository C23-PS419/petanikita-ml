import tensorflow as tf


class LeafDiseaseClassifier(tf.keras.Model):
    """A basic model for classifying leaf diseases, uses MobileNetV2 as a backbone.
    Args:
        num_classes (int): Number of classes to classify.
        input_shape (tuple): Input shape of the model.
    """
    def __init__(self, num_classes, input_shape=(512, 512, 3), *args, **kwargs):
        super().__init__(name="LeafDiseaseClassifier", *args, **kwargs)

        self.num_classes = num_classes

        self.pre_trained_model = self._get_pretrained_model(input_shape)

        for layer in self.pre_trained_model.layers:
            layer.trainable = True

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.logits = tf.keras.layers.Conv2D(1000, 3, name="logits")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.prediction = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="prediction"
        )

    def call(self, inputs, training=False, mask=None):
        x = self.pre_trained_model(inputs)
        x = self.dropout(x, training=training)
        x = self.logits(x)
        x = self.flatten(x)
        return self.prediction(x)

    def _get_pretrained_model(self, input_shape):
        return tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
        )
