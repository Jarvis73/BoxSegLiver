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

import tensorflow as tf
from tensorflow.contrib import opt as contrib_opt

from config import CustomKeys


def add_arguments(parser):
    group = parser.add_argument_group(title="Training Arguments")
    group.add_argument("--learning_rate",
                       type=float,
                       default=1e-3, help="Base learning rate for model training (default: %(default)f)")
    group.add_argument("--learning_policy",
                       type=str,
                       default="period_step",
                       choices=["custom_step", "period_step", "poly", "plateau"],
                       help="Learning rate policy for training (default: %(default)s)")
    group.add_argument("--num_of_steps",
                       type=int,
                       default=0, help="Number of steps for training")
    group.add_argument("--num_of_total_steps",
                       type=int,
                       default=1000, help="Number of total steps for training")
    group.add_argument("--lr_decay_boundaries",
                       type=int,
                       nargs="*", help="For \"custom_step\" policy. Use the specified learning rate at the "
                                       "given boundaries.")
    group.add_argument("--lr_custom_values",
                       type=float,
                       nargs="+", help="For \"custom_step\" policy. Use the specified learning rate at the "
                                       "given boundaries. "
                                       "Make sure len(lr_custom_values) - len(lr_decay_boundaries) = 1")
    group.add_argument("--lr_decay_step",
                       type=int,
                       default=1e5, help="For \"period_step\" policy. Decay the base learning rate at a fixed "
                                         "step (default: %(default)d)")
    group.add_argument("--lr_decay_rate",
                       type=float,
                       default=0.1, help="For \"period_step\" and \"plateau\" policy. Learning rate decay rate "
                                         "(default: %(default)f)")
    group.add_argument("--lr_power",
                       type=float,
                       default=0.9, help="For \"poly\" policy. Polynomial power (default: %(default)f)")
    group.add_argument("--lr_end",
                       type=float,
                       default=1e-6, help="For \"poly\" and \"plateau\" policy. The minimal end learning rate "
                                          "(default: %(default)f)")
    group.add_argument("--optimizer",
                       type=str,
                       default="Adam",
                       choices=["Adam", "Momentum", "AdamW"],
                       help="Optimizer for training (default: %(default)s)")
    group.add_argument("--lr_warm_up",
                       action="store_true", help="Warm up with a low learning rate to stabilize parameters")
    group.add_argument("--slow_start_step",
                       type=int,
                       default=1000, help="Training model with small learning rate for few steps")
    group.add_argument("--slow_start_lr",
                       type=float,
                       default=1e-4, help="Learning rate employed during slow start")
    group.add_argument("--adam_beta1", type=float)
    group.add_argument("--adam_beta2", type=float)
    group.add_argument("--adam_eps", type=float)


def get_solver_params(args, warm_up=False, slow_start_step=None, slow_start_learning_rate=None):
    optimizer_params = {}
    if args.adam_beta1:
        optimizer_params["beta1"] = args.adam_beta1
    if args.adam_beta2:
        optimizer_params["beta2"] = args.adam_beta2
    if args.adam_eps:
        optimizer_params["epsilon"] = args.adam_eps
    params = {"solver": Solver(args, optimizer_params=optimizer_params or None)}

    if warm_up:
        if slow_start_step is None or slow_start_learning_rate is None:
            raise ValueError("If warm up is True, arguments \"slow_start_step\" and "
                             "\"slow_start_learning_rate\" should be given")
        params["solver_kwargs"] = {"slow_start_step": slow_start_step,
                                   "slow_start_learning_rate": slow_start_learning_rate}
    else:
        params["solver_kwargs"] = {}

    return params


class Solver(object):
    def __init__(self, args, name=None, optimizer_params=None):
        """ Don't create ops/tensors in __init__() """
        self._args = args
        self.name = name or "Optimizer"

        # global step tensor
        # Warning: Don't create global step in __init__() function, but __call__()
        self.global_step = None

        self.learning_policy = self.args.learning_policy
        self.base_learning_rate = self.args.learning_rate
        self.learning_rate_decay_step = self.args.lr_decay_step
        self.learning_rate_decay_rate = self.args.lr_decay_rate
        self.num_of_total_steps = self.args.num_of_total_steps
        self.learning_power = self.args.lr_power
        self.end_learning_rate = self.args.lr_end
        self.learning_rate_decay_boundaries = self.args.lr_decay_boundaries
        self.learning_rate_custom_values = self.args.lr_custom_values

        self.optimizer = self.args.optimizer.lower()
        self.optimizer_params = optimizer_params

    @property
    def args(self):
        return self._args

    def _get_model_learning_rate(self, slow_start_step=0, slow_start_learning_rate=1e-4):
        """
        Gets model's learning rate.

        Computes the model's learning rate for different learning policy.

        Right now, only "custom_step", "period_step" and "poly" are supported.

        (1) The learning policy for "step" is computed as follows:
        current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)

        (2) The learning policy for "poly" is computed as follows:
        current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

        Parameters
        ----------
        slow_start_step: int
            Training model with small learning rate for the first few steps.
        slow_start_learning_rate: float
            The learning rate employed during slow start.

        Returns
        -------
        Learning rate for the specified learning policy.

        Raises
        ------
        ValueError: If learning policy is not recognized.
        """
        if self.learning_policy == 'period_step':
            learning_rate = tf.train.exponential_decay(
                self.base_learning_rate,
                self.global_step,
                self.learning_rate_decay_step,
                self.learning_rate_decay_rate,
                staircase=True)
        elif self.learning_policy == "custom_step":
            learning_rate = tf.train.piecewise_constant(
                x=self.global_step,
                boundaries=self.learning_rate_decay_boundaries,
                values=self.learning_rate_custom_values
            )
        elif self.learning_policy == 'poly':
            learning_rate = tf.train.polynomial_decay(
                self.base_learning_rate,
                self.global_step,
                self.num_of_total_steps,
                self.end_learning_rate,
                self.learning_power)
        elif self.learning_policy == "plateau":
            learning_rate, update_lr_op = plateau_decay(
                self.base_learning_rate,
                self.learning_rate_decay_rate,
                self.end_learning_rate)
        else:
            raise ValueError('Not supported learning policy.')

        # Employ small learning rate at the first few steps for warm start.
        if slow_start_step > 0:
            learning_rate = tf.where(self.global_step < slow_start_step, slow_start_learning_rate,
                                     learning_rate)
        tf.add_to_collection(CustomKeys.LEARNING_RATE, learning_rate)
        return learning_rate

    def _get_model_optimizer(self, learning_rate):
        if self.optimizer == "adam":
            optimizer_params = self.optimizer_params or {"beta1": 0.9, "beta2": 0.99}
            optimizer = tf.train.AdamOptimizer(learning_rate, **optimizer_params)
        elif self.optimizer == "momentum":
            optimizer_params = self.optimizer_params or {"momentum": 0.9}
            optimizer = tf.train.MomentumOptimizer(learning_rate, **optimizer_params)
        elif self.optimizer == "adamw":
            optimizer_params = self.optimizer_params or {"weight_decay": self._args.weight_decay_rate,
                                                         "learning_rate": learning_rate,
                                                         "beta1": 0.9, "beta2": 0.99}
            optimizer = contrib_opt.AdamWOptimizer(**optimizer_params)
        else:
            raise ValueError("Not supported optimizer: " + self.optimizer)

        return optimizer

    def __call__(self, loss, *args, **kwargs):
        lr_params = {}
        if "slow_start_step" in kwargs:
            lr_params["slow_start_step"] = kwargs.pop("slow_start_step")
        if "slow_start_learning_rate" in kwargs:
            lr_params["slow_start_learning_rate"] = kwargs.pop("slow_start_learning_rate")

        # Get global step here.
        # __call__() function will be called inside user-defined graph
        self.global_step = tf.train.get_or_create_global_step()

        with tf.variable_scope("Optimizer"):
            lr = self._get_model_learning_rate(**lr_params)
            optimizer = self._get_model_optimizer(lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=self.global_step)
            else:
                train_op = optimizer.minimize(loss, global_step=self.global_step)

        return train_op


def plateau_decay(lr, factor, min_lr):
    with tf.variable_scope("learning_rate"):
        learning_rate = tf.get_variable("value", dtype=tf.float32,
                                        initializer=lr,
                                        trainable=False,
                                        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        update_lr_op = tf.assign(learning_rate, tf.maximum(learning_rate * factor, min_lr))
        tf.add_to_collection(CustomKeys.LR_UPDATE_OPS, update_lr_op)
    return learning_rate, update_lr_op
