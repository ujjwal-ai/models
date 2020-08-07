import abc
import tensorflow as tf
from object_detection.utils import shape_utils
from object_detection.core import model



class SaccadeNetFeatureExtractor(tf.keras.Model):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 name=None,
                 channel_means=(0., 0., 0.),
                 channel_stds=(1., 1., 1.), bgr_ordering=False
                 ):
        """
        Initializes a CascadeNet feature extractor.
        :param name:  str, the name used for the underlying keras model.
        :param channel_means: A tuple of floats, denoting the mean of each channel
        which will be subtracted from it. If None or empty, we use 0s.
        :param channel_stds: A tuple of floats, denoting the standard deviation of each
        channel. Each channel will be divided by its standard deviation value.
        If None or empty, we use 1s.
        :param bgr_ordering: bool, if set will change the channel ordering to be in the
        [blue, red, green] order.
        """
        super(SaccadeNetFeatureExtractor, self).__init__(name=name)

        if not (
                isinstance(channel_means, list) or
                isinstance(channel_means, tuple) or
                channel_means is None
        ):
            raise ValueError('The argument "channel_means" must be either a '
                             'tuple or a list or None.')

        if channel_means is None:
            channel_means = (0.0, 0.0, 0.0)

        if len(channel_means) != 3:
            raise ValueError('channel_means must have a length of 3.')

        if not (
                isinstance(channel_stds, list) or
                isinstance(channel_stds, tuple) or
                channel_stds is None
        ):
            raise ValueError('The argument "channel_stds" must be either a '
                             'tuple of a list or None.')

        if channel_stds is None:
            channel_stds = (1.0, 1.0, 1.0)

        if len(channel_stds) != 3:
            raise ValueError('channel_stds must have a length of 3.')

        self._channel_means = channel_means
        self._channel_stds = channel_stds
        self._bgr_ordering = bgr_ordering

    @abc.abstractmethod
    def preprocess(self, inputs):
        #TODO: Determine if the following snippet is really needed.
        """
        if self._bgr_ordering:
            red, green, blue = tf.unstack(inputs, axis=3)
            inputs = tf.stack([blue, green, red], axis=3)

        channel_means = tf.reshape(tf.constant(self._channel_means),
                                   [1, 1, 1, -1])
        channel_stds = tf.reshape(tf.constant(self._channel_stds),
                                  [1, 1, 1, -1])
        """

        pass

    @property
    @abc.abstractmethod
    def out_stride(self):
        """The stride in the output image of the network."""
        pass

    @property
    @abc.abstractmethod
    def num_feature_outputs(self):
        """Ther number of feature outputs returned by the feature extractor."""
        pass

    @abc.abstractmethod
    def get_sub_model(self, sub_model_type):
        """Returns the underlying keras model for the given sub_model_type.

        This function is useful when we only want to get a subset of weights to
        be restored from a checkpoint.

        Args:
          sub_model_type: string, the type of sub model. Currently, CenterNet
            feature extractors support 'detection' and 'classification'.
        """
        pass


def _to_float32(x):
    return tf.cast(x, tf.float32)


def _get_shape(tensor, num_dims):
    tf.Assert(tensor.get_shape().ndims == num_dims, [tensor])
    return shape_utils.combined_static_and_dynamic_shape(tensor)


def _flatten_spatial_dimensions(batch_images):
    batch_size, height, width, channels = _get_shape(batch_images, 4)
    return tf.reshape(batch_images, [batch_size, height * width,
                                     channels])


class SaccadeNetMetaArch(model.DetectionModel):
    def __init__(self,
                 feature_extractor,
                 num_classes
                 ):
        self._feature_extractor = feature_extractor
        super(SaccadeNetMetaArch, self).__init__(
            num_classes=num_classes
        )
