# This file contains all custom elements for the NN model.
#
# Note that when the model is saved, these functions are NOT saved,
# so they must be loaded explicitly when restoring the model from file!
import numpy
import copy
from scipy.ndimage.morphology import binary_erosion
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')  #to catch FutureWarnings
    import tensorflow.keras.backend as K  # needed for custom loss function
    import tensorflow as tf

################################################################
# Custom metric

def my_r_square_metric(y_true,y_pred):
   ss_res = K.sum(K.square(y_true-y_pred))
   ss_tot = K.sum(K.square(y_true-K.mean(y_true)))
   return ( 1 - ss_res/(ss_tot + K.epsilon()) )
   
################################################################
# Custom loss

def my_mean_squared_error_noweight(y_true,y_pred):
    return K.mean( tf.square(tf.subtract(y_pred,y_true)) )

def my_mean_squared_error_weighted1(y_true,y_pred):
    return K.mean( tf.multiply( tf.exp(tf.multiply(5.0,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )

def my_mean_squared_error_weighted(weight=0.0):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.exp(tf.multiply(weight,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_linear(weight=0.0):
    # weight here is actually slope
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.add(1.0,tf.multiply(weight,y_true)) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_gaussian(weight=0.0):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( tf.exp(tf.multiply(weight,tf.square(y_true))) , tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_squared_error_weighted_genexp(weight=(1.0,0.0,0.0)):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( \
            tf.multiply( weight[0], tf.exp( tf.multiply( weight[1], tf.pow(y_true,weight[2]) ) ) ) , \
            tf.square(tf.subtract(y_pred,y_true)) ) )
    return loss

def my_mean_absolute_error_weighted_genexp(weight=(1.0,0.0,0.0)):
    def loss(y_true,y_pred):
        return K.mean( tf.multiply( \
            tf.multiply( weight[0], tf.exp( tf.multiply( weight[1], tf.pow(y_true,weight[2]) ) ) ) , \
            tf.abs(tf.subtract(y_pred,y_true)) ) )
    return loss

def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(K.maximum(input_tensor, 1e-6)) / K.log(2.)

def _erode_mask(mask_matrix, half_window_size_px):
    """Erodes binary mask.

    :param mask_matrix: See doc for `pod`.
    :param half_window_size_px: Same.
    :return: eroded_mask_matrix: Eroded version of input.
    """

    window_size_px = 2 * half_window_size_px + 1
    structure_matrix = numpy.full(
        (window_size_px, window_size_px), 1, dtype=bool
    )

    eroded_mask_matrix = binary_erosion(
        mask_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=1
    )

    return numpy.expand_dims(
        eroded_mask_matrix.astype(float), axis=(0, -1)
    )


def _apply_max_filter(input_tensor, half_window_size_px):
    """Applies maximum-filter to tensor.

    :param input_tensor: Keras tensor.
    :param half_window_size_px: Number of pixels in half of filter window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :return: output_tensor: Filtered version of `input_tensor`.
    """

    window_size_px = 2 * half_window_size_px + 1

    return K.pool2d(
        x=input_tensor, pool_mode='max',
        pool_size=(window_size_px, window_size_px), strides=(1, 1),
        padding='same', data_format='channels_last'
    )

def _create_mean_filter(half_num_rows, half_num_columns, num_channels):
    """Creates convolutional filter that computes mean.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels

    :param half_num_rows: Number of rows on either side of center.  This is
        (M - 1) / 2.
    :param half_num_columns: Number of columns on either side of center.  This
        is (N - 1) / 2.
    :param num_channels: Number of channels.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of filter weights.
    """

    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)

    return numpy.full(
        (num_rows, num_columns, num_channels, num_channels), weight,
        dtype=numpy.float32
    )

def fractions_skill_score(
        half_window_size_px, use_as_loss_function, mask_matrix,
        function_name=None, test_mode=False):
    """Fractions skill score (FSS).

    M = number of rows in grid
    N = number of columns in grid

    :param half_window_size_px: Number of pixels (grid cells) in half of
        smoothing window (on either side of center).  If this argument is K, the
        window size will be (1 + 2 * K) by (1 + 2 * K).
    :param use_as_loss_function: Boolean flag.  FSS is positively oriented
        (higher is better), but if using it as loss function, we want it to be
        negatively oriented.  Thus, if `use_as_loss_function == True`, will
        return 1 - FSS.  If `use_as_loss_function == False`, will return just
        FSS.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    # TODO(thunderhoser): Allow multiple channels.

    weight_matrix = _create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    if test_mode:
        eroded_mask_matrix = copy.deepcopy(mask_matrix)
    else:
        eroded_mask_matrix = _erode_mask(
            mask_matrix=copy.deepcopy(mask_matrix),
            half_window_size_px=half_window_size_px
        )

    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def loss(target_tensor, prediction_tensor):
        """Computes loss (fractions skill score).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        smoothed_target_tensor = K.conv2d(
            x=target_tensor, kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_prediction_tensor = K.conv2d(
            x=prediction_tensor, kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_matrix
        smoothed_prediction_tensor = (
            smoothed_prediction_tensor * eroded_mask_matrix
        )

        actual_mse = K.mean(
            (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
        )

        if use_as_loss_function:
            return actual_mse / reference_mse

        return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name

    return loss

def cross_entropy(mask_matrix, function_name=None):
    """Cross-entropy.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :return: loss: Loss function (defined below).
    """

    mask_matrix_4d = copy.deepcopy(mask_matrix)
    mask_matrix_4d = numpy.expand_dims(
        mask_matrix_4d.astype(float), axis=(0, -1)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss (cross-entropy).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        filtered_target_tensor = target_tensor * mask_matrix_4d
        filtered_prediction_tensor = prediction_tensor * mask_matrix_4d

        xentropy_tensor = (
            filtered_target_tensor * _log2(filtered_prediction_tensor) +
            (1. - filtered_target_tensor) *
            _log2(1. - filtered_prediction_tensor)
        )

        return -K.mean(xentropy_tensor)

    if function_name is not None:
        loss.__name__ = function_name

    return loss

def make_FSS_loss(mask_size):  # choose any mask size for calculating densities

    def my_FSS_loss(y_true, y_pred):

        # First: DISCRETIZE y_true and y_pred to have only binary values 0/1 
        # (or close to those for soft discretization)
        want_hard_discretization = False

        # This example assumes that y_true, y_pred have the shape (None, N, N, 1).
        
        cutoff = 0.5  # choose the cut off value for discretization

        if (want_hard_discretization):
           # Hard discretization:
           # can use that in metric, but not in loss
           y_true_binary = tf.where(y_true>cutoff, 1.0, 0.0)
           y_pred_binary = tf.where(y_pred>cutoff, 1.0, 0.0)

        else:
           # Soft discretization
           c = 10 # make sigmoid function steep
           y_true_binary = tf.math.sigmoid( c * ( y_true - cutoff ))
           y_pred_binary = tf.math.sigmoid( c * ( y_pred - cutoff ))

        # Done with discretization.

        # To calculate densities: apply average pooling to y_true.
        # Result is O(mask_size)(i,j) in Eq. (2) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (2).
        pool1 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size), strides=(1, 1), 
           padding='valid')
        y_true_density = pool1(y_true_binary);
        # Need to know for normalization later how many pixels there are after pooling
        n_density_pixels = tf.cast( (tf.shape(y_true_density)[1] * tf.shape(y_true_density)[2]) , 
           tf.float32 )

        # To calculate densities: apply average pooling to y_pred.
        # Result is M(mask_size)(i,j) in Eq. (3) of [RL08].
        # Since we use AveragePooling, this automatically includes the factor 1/n^2 in Eq. (3).
        pool2 = tf.keras.layers.AveragePooling2D(pool_size=(mask_size, mask_size),
                                                 strides=(1, 1), padding='valid')
        y_pred_density = pool2(y_pred_binary);

        # This calculates MSE(n) in Eq. (5) of [RL08].
        # Since we use MSE function, this automatically includes the factor 1/(Nx*Ny) in Eq. (5).
        MSE_n = tf.keras.losses.MeanSquaredError()(y_true_density, y_pred_density)

        # To calculate MSE_n_ref in Eq. (7) of [RL08] efficiently:
        # multiply each image with itself to get square terms, then sum up those terms.

        # Part 1 - calculate sum( O(n)i,j^2
        # Take y_true_densities as image and multiply image by itself.
        O_n_squared_image = tf.keras.layers.Multiply()([y_true_density, y_true_density])
        # Flatten result, to make it easier to sum over it.
        O_n_squared_vector = tf.keras.layers.Flatten()(O_n_squared_image)
        # Calculate sum over all terms.
        O_n_squared_sum = tf.reduce_sum(O_n_squared_vector)

        # Same for y_pred densitites:
        # Multiply image by itself
        M_n_squared_image = tf.keras.layers.Multiply()([y_pred_density, y_pred_density])
        # Flatten result, to make it easier to sum over it.
        M_n_squared_vector = tf.keras.layers.Flatten()(M_n_squared_image)
        # Calculate sum over all terms.
        M_n_squared_sum = tf.reduce_sum(M_n_squared_vector)
    
        MSE_n_ref = (O_n_squared_sum + M_n_squared_sum) / n_density_pixels
        
        # FSS score according to Eq. (6) of [RL08].
        # FSS = 1 - (MSE_n / MSE_n_ref)

        # FSS is a number between 0 and 1, with maximum of 1 (optimal value).
        # In loss functions: We want to MAXIMIZE FSS (best value is 1), 
        # so return only the last term to minimize.

        # Avoid division by zero if MSE_n_ref == 0
        # MSE_n_ref = 0 only if both input images contain only zeros.
        # In that case both images match exactly, i.e. we should return 0.
        my_epsilon = tf.keras.backend.epsilon()  # this is 10^(-7)

        if (want_hard_discretization):
           if MSE_n_ref == 0:
              return( MSE_n )
           else:
              return( MSE_n / MSE_n_ref )
        else:
           return (MSE_n / (MSE_n_ref + my_epsilon) )

    return my_FSS_loss
################################################################
# Custom categorical metrics
# Note: categorical metrics assume y_true 0-1 scaling maps to 0-60 dBZ

thr_20dbz = 0.333
thr_35dbz = 0.583
thr_50dbz = 0.833

def my_csi20_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_20dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_20dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)

def my_csi35_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_35dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_35dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)
    
def my_csi50_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_50dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_50dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return nhit / (nhit + nmis + nfal)

def my_bias20_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_20dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_20dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)

def my_bias35_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_35dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_35dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)
    
def my_bias50_metric(y_true,y_pred):
    zeros = tf.zeros_like(y_true)
    ones = tf.ones_like(y_true)
    istrue = tf.where( tf.greater(y_true,thr_50dbz),ones,zeros)
    ispred = tf.where( tf.greater(y_pred,thr_50dbz),ones,zeros)
    nottrue = tf.subtract(1.0,istrue)
    notpred = tf.subtract(1.0,ispred)
    nhit = tf.reduce_sum(tf.multiply(  istrue ,  ispred ))
    nmis = tf.reduce_sum(tf.multiply(  istrue , notpred ))
    nfal = tf.reduce_sum(tf.multiply( nottrue ,  ispred ))
    nrej = tf.reduce_sum(tf.multiply( nottrue , notpred ))
    return (nhit + nfal) / (nhit + nmis)

################################################################

