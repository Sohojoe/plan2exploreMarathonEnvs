# Copyright 2019 The Dreamer Authors. Copyright 2020 Plan2Explore Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mask(tensor, mask=None, length=None, value=0, debug=False):
  if len([x for x in (mask, length) if x is not None]) != 1:
    raise KeyError('Exactly one of mask and length must be provided.')
  with tf.name_scope('mask'):
    if mask is None:
      range_ = tf.range(tensor.shape[1].value)
      mask = range_[None, :] < length[:, None]
    batch_dims = mask.shape.ndims
    while tensor.shape.ndims > mask.shape.ndims:
      mask = mask[..., None]
    multiples = [1] * batch_dims + tensor.shape[batch_dims:].as_list()
    mask = tf.tile(mask, multiples)
    masked = tf.where(mask, tensor, value * tf.ones_like(tensor))
    if debug:
      masked = tf.check_numerics(masked, 'masked')
    return masked
