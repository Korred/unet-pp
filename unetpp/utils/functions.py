import tensorflow as tf
import keras.backend as kb


# Source: https://github.com/maju116/pyplatypus/blob/7b1706e4d6f2ef184d0e90d87226506cebb8ed1d/pyplatypus/segmentation/loss_functions.py#L91
def dice_coefficient(y_actual: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    intersection = kb.sum(tf.cast(y_actual, "float32") * y_pred)
    masks_sum = kb.sum(tf.cast(y_actual, "float32")) + kb.sum(y_pred)
    dice_coefficient = (2 * intersection + kb.epsilon()) / (masks_sum + kb.epsilon())
    return dice_coefficient


# Source: https://medium.com/@rmoklesur/reasons-to-choose-focal-loss-over-cross-entropy-5fdccb25d282
def focal_loss_(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for multi-class classification."""
    y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32
    y_pred = tf.cast(y_pred, tf.float32)  # Convert y_pred to float32
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)
