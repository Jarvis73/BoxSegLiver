# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Helper functions for running models in a distributed setting."""

import functools

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import values
from tensorflow.contrib import distribute
from tensorflow.contrib.distribute.python import mirrored_strategy


def get_distribution_strategy(distribution_strategy="default",
                              num_gpus=0,
                              num_workers=1,
                              all_reduce_alg=None):
    """Return a DistributionStrategy for running the model.

    Args:
      distribution_strategy: a string specify which distribution strategy to use.
        Accepted values are 'off', 'default', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', case insensitive. 'off' means
        not to use Distribution Strategy; 'default' means to choose from
        `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
        according to the number of GPUs and number of workers.
      num_gpus: Number of GPUs to run this model.
      num_workers: Number of workers to run this model.
      all_reduce_alg: Optional. Specify which algorithm to use when performing
        all-reduce. See tf.contrib.distribute.AllReduceCrossDeviceOps for
        available algorithms when used with `mirrored`, and
        tf.distribute.experimental.CollectiveCommunication when used with
        `multi_worker_mirrored`. If None, DistributionStrategy will choose based
        on device topology.

    Returns:
      tf.distribute.DistibutionStrategy object.
    Raises:
      ValueError: if `distribution_strategy` is 'off' or 'one_device' and
        `num_gpus` is larger than 1; or `num_gpus` is negative.
    """
    if num_gpus < 0:
        raise ValueError("`num_gpus` can not be negative.")

    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == "off":
        if num_gpus > 1 or num_workers > 1:
            raise ValueError(
                "When {} GPUs and  {} workers are specified, distribution_strategy "
                "flag cannot be set to 'off'.".format(num_gpus, num_workers))
        return None

    if distribution_strategy == "multi_worker_mirrored" or num_workers > 1:
        raise NotImplementedError
        # if all_reduce_alg not in _COLLECTIVE_COMMUNICATION_OPTIONS:
        #     raise ValueError(
        #         "When used with `multi_worker_mirrored`, valid values for "
        #         "all_reduce_alg are [`ring`, `nccl`].  Supplied value: {}".format(
        #             all_reduce_alg))
        # return tf.distribute.experimental.MultiWorkerMirroredStrategy(
        #     communication=_COLLECTIVE_COMMUNICATION_OPTIONS[all_reduce_alg])

    if (distribution_strategy == "one_device" or
            (distribution_strategy == "default" and num_gpus <= 1)):
        if num_gpus == 0:
            return distribute.OneDeviceStrategy("device:CPU:0")
        else:
            if num_gpus > 1:
                raise ValueError("`OneDeviceStrategy` can not be used for more than "
                                 "one device.")
            return distribute.OneDeviceStrategy("device:GPU:0")

    if distribution_strategy in ("mirrored", "default"):
        if num_gpus == 0:
            assert distribution_strategy == "mirrored"
            devices = ["device:CPU:0"]
        else:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]
        if all_reduce_alg:
            return ModifiedMirroredStrategy(
                devices=devices,
                cross_tower_ops=distribute.AllReduceCrossDeviceOps(
                    all_reduce_alg, num_packs=2))
        else:
            return ModifiedMirroredStrategy(devices=devices)

    if distribution_strategy == "parameter_server":
        return distribute.ParameterServerStrategy()

    raise ValueError(
        "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def per_device_batch_size(batch_size, num_gpus):
    """For multi-gpu, batch-size must be a multiple of the number of GPUs.

    Note that distribution strategy handles this automatically when used with
    Keras. For using with Estimator, we need to get per GPU batch.

    Args:
      batch_size: Global batch size to be divided among devices. This should be
        equal to num_gpus times the single-GPU batch_size for multi-gpu training.
      num_gpus: How many GPUs are used with DistributionStrategies.

    Returns:
      Batch size per device.

    Raises:
      ValueError: if batch_size is not divisible by number of devices
    """
    if num_gpus <= 1:
        return batch_size

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. Found {} '
               'GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)
    return int(batch_size / num_gpus)


class ModifiedMirroredExtended(mirrored_strategy.MirroredExtended):
    def _distribute_dataset(self, dataset_fn):
        if self._local_mode:
            # Add argument: prefetch_on_device=False
            return values.PerReplicaDataset(
                self._call_dataset_fn(dataset_fn), self._devices, prefetch_on_device=False)
        else:
            return values.MultiWorkerDataset(
                functools.partial(self._call_dataset_fn, dataset_fn),
                self._worker_devices,
                auto_shard=self._auto_shard_dataset)


class ModifiedMirroredStrategy(distribute_lib.DistributionStrategy):
    def __init__(self,
                 devices=None,
                 num_gpus=None,
                 num_gpus_per_worker=None,
                 cross_device_ops=None,
                 auto_shard_dataset=False,
                 cross_tower_ops=None):
        assert not (cross_device_ops and cross_tower_ops)
        if num_gpus is not None and num_gpus_per_worker is not None:
            raise ValueError(
                "You cannot specify both `num_gpus` and `num_gpus_per_worker`.")
        if num_gpus is None:
            num_gpus = num_gpus_per_worker
        extended = ModifiedMirroredExtended(self, devices, num_gpus,
                                            cross_device_ops or cross_tower_ops,
                                            auto_shard_dataset)
        super(ModifiedMirroredStrategy, self).__init__(extended)
