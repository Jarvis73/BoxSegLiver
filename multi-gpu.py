import tensorflow as tf
from utils import distribution_utils
import collections
from tensorflow.python.distribute import values
from tensorflow.python.util import nest


class ModelSpec(
    collections.namedtuple("ModelSpec", [
        "sum", "sub", "mul", "div"])):
    def __new__(cls, sum, sub, mul, div):
        return super(ModelSpec, cls).__new__(
            cls, sum, sub, mul, div)


num_gpu = 2
devices = ["device:GPU:%d" % i for i in range(num_gpu)]
distri = distribution_utils.ModifiedMirroredStrategy(devices=devices)


def model_fn(features):
    features = features[1]
    sum_ = features + 2
    sub_ = features - 2
    mul_ = features * 2
    div_ = {"sum": sum_, "div": tf.constant(999), "gs": tf.Variable(888)}
    res = ModelSpec(sum=sum_, sub=sub_, mul=mul_, div=div_)
    print(res)
    return res


def input_fn(mode):
    def map_fn(x):
        return {"x": x}, x
    if mode:
        x = tf.data.Dataset.range(100).map(map_fn).batch(3).prefetch(3)
    else:
        x = tf.data.Dataset.range(100, 200).map(map_fn).batch(3).prefetch(3)
    return x


def per_device_dataset(iterator, devices):
    batch = iterator.get_next()
    print(batch)
    index = {}

    def get_ith(i):
        return lambda x: x[i]

    for i, d in enumerate(devices):
        index[d] = nest.map_structure(get_ith(i), batch)

    return values.regroup(index)


def main():
    with tf.Graph().as_default() as g:
        with distri.scope():
            # Create input pipeline
            a = distri.distribute_dataset(lambda: input_fn(True))
            # b = distri.distribute_dataset(lambda: input_fn(False))
            ai = a.make_initializable_iterator()
            # bi = b.make_one_shot_iterator()
            ah = ai._iterator.string_handle()
            # bh = bi._iterator.string_handle()
            h = tf.placeholder(tf.string, [])
            it = tf.data.Iterator.from_string_handle(h,
                                                     a._dataset.output_types,
                                                     a._dataset.output_shapes,
                                                     a._dataset.output_classes)

            features = per_device_dataset(it, distri.extended._devices)
            print(features)
            # features = ai.get_next()

            # Create model
            grouped_model_spec = distri.call_for_each_replica(model_fn, features)
            print(grouped_model_spec)
            reduce_mean_sum = distri.reduce(tf.distribute.ReduceOp.MEAN, grouped_model_spec.sum)
            unwrap_sum = distri.unwrap(grouped_model_spec.sum)
    #         unwrap_sum = {key: (tf.concat(distri.unwrap(val), axis=0)
    #                             if key == "sum" else distri.unwrap(val)[0])
    #                       for key, val in grouped_model_spec.div.items()}
    #
    sess_cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_cfg.gpu_options.allow_growth = True
    with tf.Session(config=sess_cfg, graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(ai.initializer)
        # ahh, bhh = sess.run([ah, bh])
        ahh = sess.run(ah)
        d = sess.run([reduce_mean_sum, unwrap_sum], {h: ahh})
        print("data: {}, {}".format(*d))


if __name__ == "__main__":
    main()
