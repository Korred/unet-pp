import contextlib
from typing import Tuple, Optional

import tensorflow as tf

# Instead of using tensorflow.keras, we use the keras module directly
# For some reason, the tensorflow.keras cannot be resolved in the IDE (both PyCharm and VSCode)
from keras.layers import (
    Activation,
    Average,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    SpatialDropout2D,
    concatenate,
)
from keras.models import Model


class UNetPlusPlus:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        deep_supervision: bool = False,
        batch_normalization: bool = False,
        conv_activation: str = "relu",
        dropout: bool = False,
        dropout_type: Optional[str] = None,
        dropout_rate: float = 0.0,
    ):
        # For now let us assume a model with 4 levels e.g. 4 down-sampling and 4 up-sampling
        # TODO: Add support for different number of levels
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.batch_normalization = batch_normalization
        self.conv_activation = conv_activation
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate

        self.init_filters = 64

        self.model = self.build_model()

    # In order to keep the __init__ method clean, we use setters for the param validation
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value: int) -> None:
        if not isinstance(value, int) or value < 2:
            raise ValueError("Number of classes must be an integer >= 2.")

        self._num_classes = value

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value: Tuple[int, int, int]) -> None:
        # Check if input shape is a tuple of 3 positive integers
        if (
            len(value) != 3
            or not all(isinstance(i, int) for i in value)
            or not all(i > 0 for i in value)
        ):
            raise ValueError("Input shape must be a tuple of 3 integers.")

        # Check if input shape (x,y) is a tuple of 3 integers divisible by 16 (for 4 down-sampling and 4 up-sampling)
        # e.g. for a level 1 model, the input shape must be divisible by 2 (2^1)
        if value[0] % 16 != 0 or value[1] % 16 != 0:
            raise ValueError("Input shape (x,y) must be divisible by 16.")

        self._input_shape = value

    @property
    def dropout_type(self) -> str:
        return self._dropout_type

    @dropout_type.setter
    def dropout_type(self, value: str) -> None:
        if self.dropout and value not in ["simple", "spatial"]:
            raise ValueError("Invalid dropout type. Must be 'simple' or 'spatial'.")

        self._dropout_type = value

    @contextlib.contextmanager
    def options(self, options):
        old_opts = tf.config.optimizer.get_experimental_options()
        tf.config.optimizer.set_experimental_options(options)
        try:
            yield
        finally:
            tf.config.optimizer.set_experimental_options(old_opts)

    def _dropout_layer(self):
        def layer(x):
            # Workaround for:
            # E tensorflow/core/grappler/optimizers/meta_optimizer.cc:954] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4
            # @ fanin shape inmodel/dropout/dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
            # See: https://github.com/tensorflow/tensorflow/issues/34499#issuecomment-1695504189

            with self.options({"layout_optimizer": False}):
                if self.dropout_type == "simple":
                    x = Dropout(self.dropout_rate)(x)
                elif self.dropout_type == "spatial":
                    x = SpatialDropout2D(self.dropout_rate)(x)
                return x

        return layer

    def _conv_block(self, filters, kernel_size=3):
        # Mimic the function signature of other keras layers
        def layer(x):
            for _ in range(2):
                # We decided to use padding='same' and strides=(1,1) to avoid shrinking of the input image/feature maps
                x = Conv2D(filters, kernel_size, padding="same")(x)
                if self.batch_normalization:
                    x = BatchNormalization()(x)
                x = Activation(self.conv_activation)(x)

                if self.dropout:
                    x = self._dropout_layer()(x)
            return x

        return layer

    def _upconv(self, filters):
        def layer(input_layer):
            x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(
                input_layer
            )
            return x

        return layer

    def build_model(self) -> Model:
        # As opposed to the original U-Net paper:
        # - we use padding='same' to avoid shrinking of the input image/feature maps
        #   - this ensures that the output has the same spatial dimensions as the input

        # Hard-coded number of levels for now
        filters = [self.init_filters * 2**i for i in range(5)]

        model_input = Input(shape=self.input_shape)

        # Naming convention:
        # e_XY: encoder, level X, block Y
        # s_XY: skip connection, level X, block Y
        # d_XY: decoder, level X, block Y
        # $u_XY: up-convolution, level X, block Y, for $ in {s, d}
        # ep_XY: encoder pooling, level X, block Y

        # Encoder (backbone / contracting path)
        # Level 1
        e_00 = self._conv_block(filters[0])(model_input)
        ep_00 = MaxPooling2D(pool_size=(2, 2))(e_00)

        # Level 2
        e_10 = self._conv_block(filters[1])(ep_00)
        ep_10 = MaxPooling2D(pool_size=(2, 2))(e_10)

        # Level 3
        e_20 = self._conv_block(filters[2])(ep_10)
        ep_20 = MaxPooling2D(pool_size=(2, 2))(e_20)

        # Level 4
        e_30 = self._conv_block(filters[3])(ep_20)
        ep_30 = MaxPooling2D(pool_size=(2, 2))(e_30)

        # Level 5
        e_40 = self._conv_block(filters[4])(ep_30)

        # Skip connections
        # Diagonal 1
        su_10 = self._upconv(filters[0])(e_10)
        s_01 = concatenate([e_00, su_10])
        s_01 = self._conv_block(filters[0])(s_01)

        su_20 = self._upconv(filters[1])(e_20)
        s_11 = concatenate([e_10, su_20])
        s_11 = self._conv_block(filters[1])(s_11)

        su_30 = self._upconv(filters[2])(e_30)
        s_21 = concatenate([e_20, su_30])
        s_21 = self._conv_block(filters[2])(s_21)

        # Diagonal 2
        su_11 = self._upconv(filters[0])(s_11)
        s_02 = concatenate([e_00, s_01, su_11])
        s_02 = self._conv_block(filters[0])(s_02)

        su_21 = self._upconv(filters[1])(s_21)
        s_12 = concatenate([e_10, s_11, su_21])
        s_12 = self._conv_block(filters[1])(s_12)

        # Diagonal 3
        su_12 = self._upconv(filters[0])(s_12)
        s_03 = concatenate([e_00, s_01, s_02, su_12])
        s_03 = self._conv_block(filters[0])(s_03)

        # Decoder (expansion path)
        # Level 4
        du_40 = self._upconv(filters[3])(e_40)
        d_31 = concatenate([e_30, du_40])
        d_31 = self._conv_block(filters[3])(d_31)

        # Level 3
        du_31 = self._upconv(filters[2])(d_31)
        d_22 = concatenate([e_20, s_21, du_31])
        d_22 = self._conv_block(filters[2])(d_22)

        # Level 2
        du_22 = self._upconv(filters[1])(d_22)
        d_13 = concatenate([e_10, s_11, s_12, du_22])
        d_13 = self._conv_block(filters[1])(d_13)

        # Level 1
        du_13 = self._upconv(filters[0])(d_13)
        d_04 = concatenate([e_00, s_01, s_02, s_03, du_13])
        d_04 = self._conv_block(filters[0])(d_04)

        # Output layer(s)
        s_01_output = Conv2D(self.num_classes, (1, 1), activation="softmax")(s_01)
        s_02_output = Conv2D(self.num_classes, (1, 1), activation="softmax")(s_02)
        s_03_output = Conv2D(self.num_classes, (1, 1), activation="softmax")(s_03)
        d_04_output = Conv2D(self.num_classes, (1, 1), activation="softmax")(d_04)

        # By default, the model outputs the result of the last layer
        outputs = d_04_output

        # If deep supervision is enabled, the model outputs the average of the results of the skip connections and the last layer
        if self.deep_supervision:
            outputs = Average()([s_01_output, s_02_output, s_03_output, d_04_output])

        model = Model(inputs=model_input, outputs=outputs)

        return model
