from keras.constraints import Constraint
from keras import backend as BK
import tensorflow as tf


# Define custom constraint for value not being in 0 < w < 1:
class Within(Constraint):
    """Constrains the weights to be non-negative.
    """
    def __init__(self, lim_min, lim_max):
        self.lim_min = lim_min
        self.lim_max = lim_max

    def __call__(self, w):
        return w * (BK.cast(tf.math.logical_or(
            BK.greater_equal(w, self.lim_min), BK.less_equal(w, self.lim_max)
        ), BK.floatx()))

    def get_config(self):
        return {
            'lim_min': self.lim_min,
            'lim_max': self.lim_max
        }

class WeightClip(Constraint):
    def __init__(self, minimum=0.1, maximum=1.0):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, p):
        return BK.clip(p, self.minimum, self.maximum)

    def get_config(self):
        return {
            'minimum': self.minimum,
            'maximum': self.maximum
        }



def custom_loss(y_true, y_pred):
    # Normal MSE loss
    mse = BK.mean(BK.square(y_true-y_pred), axis=-1)

    # Loss that penalizes differences between sum(predictions) and sum(labels)
    #sum_constraint = K.square(K.sum(y_pred, axis=-1) - K.sum(y_true, axis=-1))

    # Clip constraint:
    clip_constraint = BK.sum(BK.square(y_pred * BK.cast(tf.math.logical_or(
            BK.less_equal(y_pred, 0.0), BK.greater_equal(y_pred, 1.0)
    ), BK.floatx())), axis=-1)

    # Overlap constraint:
    # y_pred.shape = (BATCH_SIZE, 5, )
    # LT cannot be smaller than RB:
    e = BK.cast((y_pred[:, 2:4] - y_pred[:, 0:2]), BK.floatx())
    l = BK.cast(tf.less(e, 0), BK.floatx())
    s = BK.sum(l, axis=-1)
    ol_constraint = BK.sum(BK.square(BK.sum(l*e, axis=-1) * s))


    return(mse+clip_constraint)



# t = tf.constant([[0, 2, 3, 4, 5],
#                  [0, 0, 1, 1, 15],
#                  [0, 1, 1, 0, 25],
#                  [1, 0, 0, 1, 35],
#                  [0.4, 1.2, 0, 0, 45]])
# e = BK.cast((t[:, 2:4] - t[:, 0:2]), BK.floatx())
# display(e)
# l = BK.cast(tf.less(e, 0), BK.floatx())
# s = BK.sum(l, axis=-1)
# display(l)
# v= BK.sum(l*e, axis=-1) * s
# BK.sum(BK.square(v), axis=-1)


def ImageAugmentator():
    pass