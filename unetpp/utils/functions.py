import tensorflow as tf
import keras.backend as kb

# Source: https://github.com/maju116/pyplatypus/blob/7b1706e4d6f2ef184d0e90d87226506cebb8ed1d/pyplatypus/segmentation/loss_functions.py#L91
def dice_coefficient(y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    intersection = kb.sum(tf.cast(y_actual, 'float32') * y_pred)
    masks_sum = kb.sum(tf.cast(y_actual, 'float32')) + kb.sum(y_pred)
    dice_coefficient = (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())
    return dice_coefficient