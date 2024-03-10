import tensorflow as tf
import keras.backend as kb


# Source: https://github.com/maju116/pyplatypus/blob/7b1706e4d6f2ef184d0e90d87226506cebb8ed1d/pyplatypus/segmentation/loss_functions.py#L91
def dice_coefficient(y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    intersection = kb.sum(tf.cast(y_actual, "float32") * y_pred)
    masks_sum = kb.sum(tf.cast(y_actual, "float32")) + kb.sum(y_pred)
    dice_coefficient = (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())
    return dice_coefficient


# # focal loss
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32
        y_pred = tf.cast(y_pred, tf.float32)  # Convert y_pred to float32
        bce = kb.binary_crossentropy(y_true, y_pred)

        y_pred = kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon())
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
        modulating_factor = kb.pow((1 - p_t), gamma)

        # compute the final loss and return
        return kb.mean(alpha_factor * modulating_factor * bce, axis=-1)

    return focal_crossentropy
