import tensorflow as tf

# Batch Normalization
from tensorflow.contrib import slim


def get_normalization_fn(x, norm_type, is_train, G=16, esp=1e-5, scope=None):
    """
    :param x:input data with shap of[batch,height,width,channel]
    :param is_train:flag of normalization,True is training,False is Testing
    :param norm_type:normalization type:support"batch","group","None"
    :param G:in group normalization,channel is seperated with group number(G)
    :param esp:Prevent divisor from being zero
    :param scope:normalization scope
    :return:
    """
    if norm_type is None:
        output = x
    elif norm_type == 'batch':
        output = tf.layers.batch_normalization(x, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                               training=is_train)
    elif norm_type == "group":
        output = group_normalization(x, G, esp, '' if scope is None else scope)
    elif norm_type == 'instance':
        output = slim.instance_norm(x)
    else:
        raise Exception("Invalid normalization function")
    return output


def group_normalization(x, G, esp, scope):
    with tf.variable_scope('group_normalization'):
        # tranpose:[bs,z,h,w,c]to[bs,c,z,h,w]following the paper
        x = tf.transpose(x, [0, 4, 1, 2, 3])
        N, C, Z, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, Z, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        gama = tf.get_variable(scope + '_group_gama', [C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(scope + '_group_beta', [C], initializer=tf.constant_initializer(0.0))
        gama = tf.reshape(gama, [1, C, 1, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1, 1])
        output = tf.reshape(x, [-1, C, Z, H, W]) * gama + beta
        # tranpose:[bs,c,z,h,w]to[bs,z,h,w,c]following the paper
        output = tf.transpose(output, [0, 2, 3, 4, 1])
        return output
