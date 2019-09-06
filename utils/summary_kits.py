import tqdm
import tensorflow as tf
from io import BytesIO
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summary_scalar(writer, iter_, tags, values):
    """ Summary a scalar in tensorboard manually.

    Params
    ------
    `writer`: a tf.summary.FileWriter instance
    `iter`: a integer to denote current iteration
    `tags`: tag of the scalar, multi-level tag should be seperated by `/`.
    You can pass a single tag or a list of tags.
    `values`: scalar value to be summaried.

    Note: `tags` and `values` should have same length(i.e. both single entry
    or a list of entries)
    """

    if not isinstance(tags, (str, list, tuple)):
        raise TypeError("tags should have type of (str, list, tuple), but got {}".format(type(tags)))
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(values, (float, int, list, tuple)):
        raise TypeError("values should have type of (float, int, list, tuple), but got {}".format(type(values)))
    if isinstance(values, (float, int)):
        values = [values]

    all_value = []
    for tag, value in zip(tags, values):
        all_value.append(tf.Summary.Value(tag=tag, simple_value=value))

    summary_value = tf.Summary(value=all_value)
    writer.add_summary(summary_value, int(iter_))

    return


def summary_image(writer, iter_, tag, images, max_outputs=3):
    """ Summary a batch images in tensorboard manually.

    Params
    ------
    `writer`: a tf.summary.FileWriter instance
    `iter`: a integer to denote current iteration
    `tag`: tag of the image, details please reference `tf.summary.image`
    `images`: 4D np.ndarray with shape [batch_size, height, width, channels]
    `max_outputs`: Max number of batch elements to generate images for.
    """

    all_value = []
    for i, image in enumerate(images):
        buffer = BytesIO()
        plt.imsave(buffer, image, format="png")
        image_sum_obj = tf.Summary.Image(height=image.shape[0], width=image.shape[1], colorspace=3,
                                         encoded_image_string=buffer.getvalue())
        all_value.append(tf.Summary.Value(tag="{:s}/image/{:d}".format(tag, i), image=image_sum_obj))
        if i + 1 >= max_outputs:
            break

    summary_value = tf.Summary(value=all_value)
    writer.add_summary(summary_value, int(iter_))

    return


def change_summary_prefix(event_file, write_dir, new_prefix=None, remove_prefix=False,
                          keep_fields=("simple_value",)):
    write_dir = Path(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    writer = tf.summary.FileWriter(write_dir)

    def modify(value):
        if remove_prefix:
            new_tag = "/".join(value.tag.split("/")[1:])
            value.tag = new_tag
        elif new_prefix:
            new_tag = "/".join([new_prefix] + value.tag.split("/")[1:])
            value.tag = new_tag
        # if value.WhichOneof("value") == "simple_value":
        #     value.simple_value += 0.1
        return value

    total = 0
    for _ in tf.train.summary_iterator(event_file):
        total += 1

    for event in tqdm.tqdm(tf.train.summary_iterator(event_file), total=total):
        event_type = event.WhichOneof("what")
        if event_type != "summary":
            writer.add_event(event)
        else:
            wall_time = event.wall_time
            step = event.step
            filtered_values = [modify(value) if new_prefix or remove_prefix else value
                               for value in event.summary.value if value.WhichOneof("value") in keep_fields]
            summary = tf.Summary(value=filtered_values)
            filtered_event = tf.summary.Event(summary=summary, wall_time=wall_time, step=step)
            writer.add_event(filtered_event)
    writer.close()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, help="Event file")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-p", "--new_prefix", type=str, help="New prefix")
    parser.add_argument("-r", "--remove_prefix", action="store_true", help="Remove current prefix")
    parser.add_argument("-k", "--keep_fields", type=str, nargs="+", default=["simple_value"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    import numpy as np
    a = np.random.rand(2, 64, 64, 3) * 255
    a = a.astype(np.uint8)
    writer = tf.summary.FileWriter("D:/Library/Downloads")
    summary_image(writer, 0, "AAA", a)
    writer.close()
