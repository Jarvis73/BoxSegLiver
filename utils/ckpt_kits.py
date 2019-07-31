# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# =================================================================================

import argparse
import tensorflow as tf


def list_vars_shape(path):
    reader = tf.train.load_checkpoint(path)
    variables = reader.get_variable_to_shape_map()
    for k in sorted(variables):
        print(k, variables[k])


def list_vars_dtype(path):
    reader = tf.train.load_checkpoint(path)
    variables = reader.get_variable_to_dtype_map()
    for k in sorted(variables):
        print(k, variables[k])


def ckpt_vars_rename(input_, output=None, replace_from=(), replace_to=(), add_prefix=None):
    assert len(replace_from) == len(replace_to), (len(replace_from), len(replace_to))

    reader = tf.train.load_checkpoint(input_)
    variables = reader.get_variable_to_shape_map()

    replace_to = ["" if x == "empty" else x for x in replace_to]
    for k in sorted(variables):
        new_name = k
        if len(replace_from) > 0:
            for from_, to_ in zip(replace_from, replace_to):
                new_name = new_name.replace(from_, to_)
        if add_prefix:
            new_name = add_prefix + new_name

        print("%s --> %s" % (k, new_name))
        if output:
            var = reader.get_tensor(k)
            tf.Variable(var, name=new_name)

    if output:
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, output)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str,
                        choices=['list_vars_shape', 'list_vars_dtype', 'ckpt_vars_rename'],
                        help="Execute which function.")
    parser.add_argument("input", type=str, help="Input checkpoint file")
    parser.add_argument("-o", "--output", type=str, help="Output checkpoint file")
    parser.add_argument("-f", "--fro", type=str, nargs="+", help="Replace from")
    parser.add_argument("-t", "--to", type=str, nargs="+", help="Replace to, length must be equal to --fro")
    parser.add_argument("-p", "--prefix", type=str, help="Add prefix")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    if args.command == "list_vars_shape":
        list_vars_shape(args.input)
    elif args.command == "list_vars_dtype":
        list_vars_dtype(args.input)
    elif args.command == "ckpt_vars_rename":
        if (args.fro and args.to and len(args.fro) != len(args.to)) or \
                (args.fro and not args.to) or (not args.fro and args.to):
            raise ValueError("--fro and --to must be given equal number of arguments")
        ckpt_vars_rename(args.input, args.output, args.fro or (), args.to or (), args.prefix)
