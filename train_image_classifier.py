# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MODIFIED to use tf_slim and to add saving-every-n-steps functionality.
Generic training script that trains a model using a given dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tf_slim_training
import efficient_second as es

from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import slim
from tensorflow.python import app
from tensorflow.compat.v1.train import Scaffold
from tensorflow.python.training import saver as tf_saver
from tensorflow.core.protobuf import saver_pb2

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

# ADDED: Do not force explicit placement. In particular for ops for the GPU.
tf.config.set_soft_device_placement(True)

app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
app.flags.DEFINE_float(
    'warmup_epochs', 0,
    'Linearly warmup learning rate from 0 to learning_rate over this '
    'many epochs.')

app.flags.DEFINE_integer('num_clones', 1,
                         'Number of model clones to deploy. Note For '
                         'historical reasons loss from all clones averaged '
                         'out and learning rate decay happen per clone '
                         'epochs')

app.flags.DEFINE_boolean('clone_on_cpu', False,
                         'Use CPUs to deploy clones.')

app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

app.flags.DEFINE_integer(
    'save_summaries_secs', None,  # 600
    'The frequency with which summaries are saved, in seconds.')

app.flags.DEFINE_integer(
    'save_interval_secs', None,  # 600
    'The frequency with which the model is saved, in seconds.')

app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                       'The learning rate power.')

app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

app.flags.DEFINE_float('rmsprop_decay', 0., 'Decay term for RMSProp.')

app.flags.DEFINE_integer(
    'quantize_delay', -1,
    'Number of steps to start quantized training. Set to -1 would disable '
    'quantized training.')

#######################
# Learning Rate Flags #
#######################

app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays. Note: this flag counts '
    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
    'each clone will go over full epoch individually, but replicas will go '
    'once across all replicas.')

app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

app.flags.DEFINE_integer('max_number_of_steps', None,
                         'The maximum number of training steps.')

app.flags.DEFINE_bool('use_grayscale', False,
                      'Whether to convert input images to grayscale.')

#####################
# Fine-Tuning Flags #
#####################

app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

################
# My own Flags #
################

app.flags.DEFINE_integer(
    'save_summaries_steps', None, 'Saves summaries every n steps.')

app.flags.DEFINE_integer(
    'save_checkpoints_steps', None, 'Saves model ckpt every n steps.')

app.flags.DEFINE_float(
    'eso_tau', 1.,
    'Tikhonov regularization factor for ESO.')

app.flags.DEFINE_float(
    'eso_cg_tol', 1e-5,
    'Conjugate Gradients Convergence tolerance for ESO.')

app.flags.DEFINE_integer(
    'eso_max_iter', 20,
    'Conjugate Gradients maximum number of iterations for ESO.')

app.flags.DEFINE_boolean(
    'profiling_enabled', True,
    'If profiling information should be collected during the training.')

app.flags.DEFINE_integer(
    'profile_every_n_steps', 100,
    'Every n-th step profile traces are saved. Only if profiling_enabled=True.')

app.flags.DEFINE_string(
    'profile_dir', None,
    'Path to directory to write profiling logs to.'
    'By default, None would mean profile logs are written to train_dir.')

FLAGS = app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  # Note: when num_clones is > 1, this will actually have each clone to go
  # over each epoch FLAGS.num_epochs_per_decay times. This is different
  # behavior from sync replicas and is expected to produce different results.
  steps_per_epoch = num_samples_per_epoch / FLAGS.batch_size
  if FLAGS.sync_replicas:
    steps_per_epoch /= FLAGS.replicas_to_aggregate
  
  decay_steps = int(steps_per_epoch * FLAGS.num_epochs_per_decay)
  
  if FLAGS.learning_rate_decay_type == 'exponential':
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.learning_rate_decay_factor,
        staircase=True,
        name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    learning_rate = tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    learning_rate = tf.train.polynomial_decay(
        FLAGS.learning_rate,
        global_step,
        decay_steps,
        FLAGS.end_learning_rate,
        power=1.0,
        cycle=False,
        name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                     FLAGS.learning_rate_decay_type)
  
  if FLAGS.warmup_epochs:
    warmup_lr = (
        FLAGS.learning_rate * tf.cast(global_step, tf.float32) /
        (steps_per_epoch * FLAGS.warmup_epochs))
    learning_rate = tf.minimum(warmup_lr, learning_rate)
  return learning_rate


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif FLAGS.optimizer == 'second':
    optimizer = es.EHNewtonOptimizer(
        learning_rate,
        tau=FLAGS.eso_tau,
        cg_tol=FLAGS.eso_cg_tol,
        max_iter=FLAGS.eso_max_iter)
  else:
    raise ValueError('Optimizer [%s] was not recognized' % FLAGS.optimizer)
  return optimizer


def _assign_from_checkpoint_fn(model_path,
                               var_list,
                               ignore_missing_vars=False,
                               reshape_variables=False):
  """COPIED from tf_slim/ops/variables.py
  (the file imported via `from tf_slim.ops import variables`)
  MODIFIED the returned callback to take two arguments conforming with Scaffold
  Returns a function that assigns specific variables from a checkpoint.

  If ignore_missing_vars is True and no variables are found in the checkpoint
  it returns None.

  Args:
    model_path: The full path to the model checkpoint. To get latest checkpoint
      use `model_path = tf.train.latest_checkpoint(checkpoint_dir)`
    var_list: A list of `Variable` objects or a dictionary mapping names in the
      checkpoint to the corresponding variables to initialize. If empty or
      `None`, it would return `no_op(), None`.
    ignore_missing_vars: Boolean, if True it would ignore variables missing in
      the checkpoint with a warning instead of failing.
    reshape_variables: Boolean, if True it would automatically reshape variables
      which are of different shape then the ones stored in the checkpoint but
      which have the same number of elements.

  Returns:
    A function that takes a two arguments,
    a tf.compat.v1.train.Scaffold and a `tf.compat.v1.Session`, that applies the
    assignment operation. If no matching variables were found in the checkpoint
    then `None` is returned.

  Raises:
    ValueError: If var_list is empty.
  """
  if not var_list:
    raise ValueError('var_list cannot be empty')
  if ignore_missing_vars:
    reader = tf.train.NewCheckpointReader(model_path)
    if isinstance(var_list, dict):
      var_dict = var_list
    else:
      var_dict = {var.op.name: var for var in var_list}
    available_vars = {}
    for var in var_dict:
      if reader.has_tensor(var):
        available_vars[var] = var_dict[var]
      else:
        tf.logging.warning('Variable %s missing in checkpoint %s', var, model_path)
    var_list = available_vars
  if var_list:
    saver = tf_saver.Saver(
        var_list,
        reshape=reshape_variables,
        write_version=saver_pb2.SaverDef.V1)
    
    def callback(scaffold, session):  # MODIFIED: new `scaffold` argument
      saver.restore(session, model_path)
    
    return callback
  else:
    tf.logging.warning('No Variables to restore')
    return None


def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None
  
  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.train_dir)
    return None
  
  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
  
  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        break
    else:
      variables_to_restore.append(var)
  
  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path
  
  tf.logging.info('Fine-tuning from %s' % checkpoint_path)
  
  # return slim.assign_from_checkpoint_fn(
  #    checkpoint_path,
  #    variables_to_restore,
  #    ignore_missing_vars=FLAGS.ignore_missing_vars)
  # CHANGED above to call my modified version of `_assign_from_checkpoint_fn()`
  return _assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)


def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
  
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.worker_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)
    
    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()
    
    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    
    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    
    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True,
        use_grayscale=FLAGS.use_grayscale)
    
    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      provider = slim.dataset_data_provider.DatasetDataProvider(
          dataset,
          num_readers=FLAGS.num_readers,
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
      [image, label] = provider.get(['image', 'label'])
      label -= FLAGS.labels_offset
      
      train_image_size = FLAGS.train_image_size or network_fn.default_image_size
      
      image = image_preprocessing_fn(image, train_image_size, train_image_size)
      
      images, labels = tf.train.batch(
          [image, label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
      labels = slim.one_hot_encoding(
          labels, dataset.num_classes - FLAGS.labels_offset)
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)
    
    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)
      
      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        slim.losses.softmax_cross_entropy(
            end_points['AuxLogits'], labels,
            label_smoothing=FLAGS.label_smoothing, weights=0.4,
            scope='aux_loss')
      slim.losses.softmax_cross_entropy(
          logits, labels, label_smoothing=FLAGS.label_smoothing, weights=1.0)
      return end_points
    
    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    
    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
    
    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))
    
    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))
    
    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))
    
    #################################
    # Configure the moving averages #
    #################################
    if FLAGS.moving_average_decay:
      moving_average_variables = slim.get_model_variables()
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, global_step)
    else:
      moving_average_variables, variable_averages = None, None
    
    if FLAGS.quantize_delay >= 0:
      contrib_quantize.create_training_graph(quant_delay=FLAGS.quantize_delay)
    
    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))
    
    if FLAGS.sync_replicas:
      # If sync_replicas is enabled, the averaging will be done in the chief
      # queue runner.
      optimizer = tf.train.SyncReplicasOptimizer(
          opt=optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.worker_replicas,
          variable_averages=variable_averages,
          variables_to_average=moving_average_variables)
    elif FLAGS.moving_average_decay:
      # Update ops executed locally by trainer.
      update_ops.append(variable_averages.apply(moving_average_variables))
    
    # Variables to train.
    variables_to_train = _get_variables_to_train()
    
    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))
    
    # MODIFIED: Gradient update ops removed:
    # tf_slim.training.train adds those updates implicitly
    
    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))
    
    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
    #########################################################################
    # Everything below replaces the old training loop and is COMPLETELY NEW.#
    #########################################################################
    
    # Create tf_slim.training conforming training op
    train_op = tf_slim_training.create_train_op(
        total_loss=total_loss,
        optimizer=optimizer,
        variables_to_train=variables_to_train,
        summarize_gradients=True)
    
    # Add a Hook to stop after specified number of steps
    stop_after_n = tf.python.train.StopAtStepHook(
        num_steps=FLAGS.max_number_of_steps)
    
    # Add a Hook to enable profiling
    if FLAGS.profiling_enabled:
      profile_dir = FLAGS.profile_dir or FLAGS.train_dir
      profile_every_n = tf.python.train.ProfilerHook(
          save_steps=FLAGS.profile_every_n_steps,
          save_secs=None,
          output_dir=profile_dir,
          show_dataflow=True,
          show_memory=False)
      hooks = [stop_after_n, profile_every_n]
    else:
      hooks = [stop_after_n]
    
    # Need a scaffold to init model from checkpoint
    scaffold = Scaffold(init_fn=_get_init_fn())
    
    sum_steps = None if FLAGS.save_summaries_secs is not None else FLAGS.save_summaries_steps
    cpt_steps = None if FLAGS.save_interval_secs is not None else FLAGS.save_checkpoints_steps
    sum_secs = None if FLAGS.save_summaries_steps is not None else FLAGS.save_summaries_secs
    cpt_secs = None if FLAGS.save_checkpoints_steps is not None else FLAGS.save_interval_secs
    
    ##################################################################
    # Kicks off the training. MODIFIED to add logging-every-n-steps. #
    ##################################################################
    
    tf_slim_training.train(
        train_op=train_op,
        logdir=FLAGS.train_dir,
        scaffold=scaffold,
        hooks=hooks,
        save_summaries_steps=sum_steps,
        save_checkpoint_steps=cpt_steps,
        save_summaries_secs=sum_secs,
        save_checkpoint_secs=cpt_secs)


if __name__ == '__main__':
  app.run()
