"""Writes metadata and label file to the image classifier models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tflite_support import flatbuffers
from tflite_support import metadata_schema_py_generated as _metadata_fb
from tflite_support import metadata as _metadata

# pylint: enable=g-direct-tensorflow-import

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "model_file", None, "Path and file name to the TFLite model file."
    )
    flags.DEFINE_string("label_file", None, "Path to the label file.")
    flags.DEFINE_string(
        "export_directory",
        "converted_model_metadata/",
        "Path to save the TFLite model files with metadata.",
    )
    flags.mark_flag_as_required("model_file")
    flags.mark_flag_as_required("label_file")


class ModelSpecificInfo(object):
    def __init__(
        self,
        name,
        version,
        image_width,
        image_height,
        image_min,
        image_max,
        mean,
        std,
        num_classes,
        author,
    ):
        self.name = name
        self.version = version
        self.image_width = image_width
        self.image_height = image_height
        self.image_min = image_min
        self.image_max = image_max
        self.mean = mean
        self.std = std
        self.num_classes = num_classes
        self.author = author


_MODEL_INFO = {
    "rldc_mobilenet_v1_1_default_1.tflite": ModelSpecificInfo(
        name="Rice Leaf Disease Classifier",
        version="v1",
        image_width=224,
        image_height=224,
        image_min=0,
        image_max=255,
        mean=[127.5],
        std=[127.5],
        num_classes=4,
        author="Risang Baskoro",
    )
}


class MetadataPopulatorForImageClassifier(object):
    def __init__(self, model_file, model_info, label_file_path):
        self.model_file = model_file
        self.model_info = model_info
        self.label_file_path = label_file_path
        self.metadata_buf = None

    def populate(self):
        self._create_metadata()
        self._populate_metadata()

    def _create_metadata(self):
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_info.name
        model_meta.description = (
            f"Identify rice leafs disease from {self.model_info.num_classes} classes"
        )

        model_meta.version = self.model_info.version
        model_meta.author = self.model_info.author
        model_meta.license = (
            "MIT License. See https://opensource.org/licenses/MIT for more information."
        )

        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "image"
        input_meta.description = f"Input image to be classified. The expected image is {self.model_info.image_width} x {self.model_info.image_height}, \
                                with three channels (red, blue, and green) per pixel. Each value in the tensor is a single byte between \
                                {self.model_info.image_min} and {self.model_info.image_max}."
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = (
            _metadata_fb.ColorSpaceType.RGB
        )
        input_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.ImageProperties
        )
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions
        )
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = self.model_info.mean
        input_normalization.options.std = self.model_info.std
        input_meta.processUnits = [input_normalization]
        input_stats = _metadata_fb.StatsT()
        input_stats.max = [self.model_info.image_max]
        input_stats.min = [self.model_info.image_min]
        input_meta.stats = input_stats

        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = (
            f"Probabilities of the {self.model_info.num_classes} labels respectively."
        )
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.content_properties = _metadata_fb.FeaturePropertiesT()
        output_meta.content.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties
        )
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for objects that the model can recognize."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
        )
        self.metadata_buf = b.Output()

    def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([self.label_file_path])
        populator.populate()


def main(_):
    model_file = FLAGS.model_file
    model_basename = os.path.basename(model_file)
    if model_basename not in _MODEL_INFO:
        raise ValueError(f"The model info for, {model_basename}, is not defined yet.")

    export_model_path = os.path.join(FLAGS.export_directory, model_basename)

    tf.io.gfile.copy(model_file, export_model_path, overwrite=True)

    populator = MetadataPopulatorForImageClassifier(
        export_model_path, _MODEL_INFO.get(model_basename), FLAGS.label_file
    )
    populator.populate()

    displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
    export_json_file = os.path.join(
        FLAGS.export_directory, os.path.splitext(model_basename)[0] + ".json"
    )
    json_file = displayer.get_metadata_json()
    with open(export_json_file, "w") as f:
        f.write(json_file)

    print("Finished populating metadata and associated file to the model:")
    print(model_file)
    print("The metadata json file has been saved to:")
    print(export_json_file)
    print("The associated file that has been been packed to the model is:")
    print(displayer.get_packed_associated_file_list())


if __name__ == "__main__":
    define_flags()
    app.run(main)
