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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from plan2explore.tools import count_dataset
from plan2explore.tools import gif_summary
from plan2explore.tools import image_strip_summary
from plan2explore.tools import shape as shapelib


def plot_summary(titles, lines, labels, name):
  # Use control dependencies to ensure that only one plot summary is executed
  # at a time, since matplotlib is not thread safe.
  def body_fn(lines):
    fig, axes = plt.subplots(
        nrows=len(titles), ncols=1, sharex=True, sharey=False,
        squeeze=False, figsize=(6, 3 * len(lines)))
    axes = axes[:, 0]
    for index, ax in enumerate(axes):
      ax.set_title(titles[index])
      for line, label in zip(lines[index], labels[index]):
        ax.plot(line, label=label)
      if any(labels[index]):
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image
  image = tf.py_func(body_fn, (lines,), tf.uint8)
  image = image[None]
  summary = tf.summary.image(name, image)
  return summary


def data_summaries(
    data, postprocess_fn, histograms=False, max_batch=6, name='data'):
  summaries = []
  with tf.variable_scope(name):
    if histograms:
      for key, value in data.items():
        if key in ('image',):
          continue
        summaries.append(tf.summary.histogram(key, data[key]))
    image = data['image'][:max_batch]
    if postprocess_fn:
      image = postprocess_fn(image)
    summaries.append(image_strip_summary.image_strip_summary('image', image))
  return summaries


def dataset_summaries(directory, name='dataset'):
  summaries = []
  with tf.variable_scope(name):
    episodes = count_dataset.count_dataset(directory)
    summaries.append(tf.summary.scalar('episodes', episodes))
  return summaries


def state_summaries(
    cell, prior, posterior, histograms=False, name='state'):
  summaries = []
  divergence = cell.divergence_from_states(posterior, prior)
  prior = cell.dist_from_state(prior)
  posterior = cell.dist_from_state(posterior)
  prior_entropy = prior.entropy()
  posterior_entropy = posterior.entropy()
  nan_to_num = lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x)
  with tf.variable_scope(name):
    if histograms:
      summaries.append(tf.summary.histogram(
          'prior_entropy_hist', nan_to_num(prior_entropy)))
    summaries.append(tf.summary.scalar(
        'prior_entropy', tf.reduce_mean(prior_entropy)))
    summaries.append(tf.summary.scalar(
        'prior_std', tf.reduce_mean(prior.stddev())))
    if histograms:
      summaries.append(tf.summary.histogram(
          'posterior_entropy_hist', nan_to_num(posterior_entropy)))
    summaries.append(tf.summary.scalar(
        'posterior_entropy', tf.reduce_mean(posterior_entropy)))
    summaries.append(tf.summary.scalar(
        'posterior_std', tf.reduce_mean(posterior.stddev())))
    summaries.append(tf.summary.scalar(
        'divergence', tf.reduce_mean(divergence)))
  return summaries


def dist_summaries(dists, obs, name='dist_summaries'):
  summaries = []
  with tf.variable_scope(name):
    for name, dist in dists.items():
      mode = tf.cast(dist.mode(), tf.float32)
      mode_mean, mode_var = tf.nn.moments(mode, list(range(mode.shape.ndims)))
      mode_std = tf.sqrt(mode_var)
      summaries.append(tf.summary.scalar(name + '_mode_mean', mode_mean))
      summaries.append(tf.summary.scalar(name + '_mode_std', mode_std))
      std = dist.stddev()
      std_mean, std_var = tf.nn.moments(std, list(range(std.shape.ndims)))
      std_std = tf.sqrt(std_var)
      summaries.append(tf.summary.scalar(name + '_std_mean', std_mean))
      summaries.append(tf.summary.scalar(name + '_std_std', std_std))
      if hasattr(dist, 'distribution') and hasattr(
          dist.distribution, 'distribution'):
        inner = dist.distribution.distribution
        inner_std = tf.reduce_mean(inner.stddev())
        summaries.append(tf.summary.scalar(name + '_inner_std', inner_std))
      if name in obs:
        log_prob = tf.reduce_mean(dist.log_prob(obs[name]))
        summaries.append(tf.summary.scalar(name + '_log_prob', log_prob))
        abs_error = tf.reduce_mean(tf.abs(mode - obs[name]))
        summaries.append(tf.summary.scalar(name + '_abs_error', abs_error))
  return summaries


def image_summaries(dist, target, name='image', max_batch=6):
  summaries = []
  with tf.variable_scope(name):
    image = dist.mode()[:max_batch]
    target = target[:max_batch]
    error = ((image - target) + 1) / 2
    # empty_frame = 0 * target[:max_batch, :1]
    # change = tf.concat([empty_frame, image[:, 1:] - image[:, :-1]], 1)
    # change = (change + 1) / 2
    summaries.append(image_strip_summary.image_strip_summary(
        'prediction', image))
    # summaries.append(image_strip_summary.image_strip_summary(
    #     'change', change))
    # summaries.append(image_strip_summary.image_strip_summary(
    #     'error', error))
    # Concat prediction and target vertically.
    frames = tf.concat([target, image, error], 2)
    # Stack batch entries horizontally.
    frames = tf.transpose(frames, [1, 2, 0, 3, 4])
    s = shapelib.shape(frames)
    frames = tf.reshape(frames, [s[0], s[1], s[2] * s[3], s[4]])
    summaries.append(gif_summary.gif_summary(
        'gif', frames[None], max_outputs=1, fps=20))
  return summaries


def objective_summaries(objectives, name='objectives'):
  summaries = []
  with tf.variable_scope(name):
    for objective in objectives:
      summaries.append(tf.summary.scalar(objective.name, objective.value))
  return summaries


def prediction_summaries(dists, data, state, name='state'):
  summaries = []
  with tf.variable_scope(name):
    # Predictions.
    log_probs = {}
    for key, dist in dists.items():
      if key in ('image',):
        continue
      if key not in data:
        continue
      # We only look at the first example in the batch.
      log_prob = dist.log_prob(data[key])[0]
      prediction = dist.mode()[0]
      truth = data[key][0]
      plot_name = key
      # Ensure that there is a feature dimension.
      if prediction.shape.ndims == 1:
        prediction = prediction[:, None]
        truth = truth[:, None]
      prediction = tf.unstack(tf.transpose(prediction, (1, 0)))
      truth = tf.unstack(tf.transpose(truth, (1, 0)))
      lines = list(zip(prediction, truth))
      titles = ['{} {}'.format(key.title(), i) for i in range(len(lines))]
      labels = [['Prediction', 'Truth']] * len(lines)
      plot_name = '{}_trajectory'.format(key)
      # The control dependencies are needed because rendering in matplotlib
      # uses global state, so rendering two plots in parallel interferes.
      with tf.control_dependencies(summaries):
        summaries.append(plot_summary(titles, lines, labels, plot_name))
      log_probs[key] = log_prob
    log_probs = sorted(log_probs.items(), key=lambda x: x[0])
    titles, lines = zip(*log_probs)
    titles = [title.title() for title in titles]
    lines = [[line] for line in lines]
    labels = [[None]] * len(titles)
    plot_name = 'logprobs'
    with tf.control_dependencies(summaries):
      summaries.append(plot_summary(titles, lines, labels, plot_name))
  return summaries
