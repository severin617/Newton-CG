# TODO licence

import tensorflow as tf
import trusting_hooks as th
import trusting_session as ts

from tensorflow.python import app

app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

app.flags.DEFINE_integer(
    'save_summaries_secs', None,  # 600
    'The frequency with which summaries are saved, in seconds.')

app.flags.DEFINE_integer(
    'save_interval_secs', None,  # 600
    'The frequency with which the model is saved, in seconds.')

######################
# Optimization Flags #
######################

# We use the config from the first training that we restore.

#######################
# Learning Rate Flags #
#######################

# We use the config from the first training that we restore.

#######################
# Dataset Flags #
#######################

app.flags.DEFINE_integer('max_number_of_steps', None,
                         'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

# This has been configured before the first training.
# On restoring, we continue with the same config.

################
# My own Flags #
################

app.flags.DEFINE_integer(
    'save_summaries_steps', None, 'Saves summaries every n steps.')

app.flags.DEFINE_integer(
    'save_checkpoints_steps', None, 'Saves model ckpt every n steps.')

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


def train(logdir,
          save_checkpoint_secs=600,
          save_checkpoint_steps=None,
          save_summaries_secs=None,
          save_summaries_steps=100,
          max_wait_secs=7200):
  """COPIED and ADAPTED from train() in tf_slim_training.py
  Runs the training loop.

  Args:
    logdir: The directory where the graph and checkpoints are saved.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_checkpoint_steps: ADDED to provide saving-every-n-steps functionality.
    save_summaries_secs: ADDED to provide saving-every-n-secs functionality
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If
      `save_summaries_steps` is set to `None`, then the default summary saver
      isn't used.
    max_wait_secs: Maximum time workers should wait for the session to become
      available. This should be kept relatively short to help detect incorrect
      code, but sometimes may need to be increased if the chief takes a while to
      start up.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `logdir` is `None` and either `save_checkpoint_secs` or
    `save_summaries_steps` are `None.
  """
  if logdir is None:
    if save_summaries_steps:
      raise ValueError(
          'logdir cannot be None when save_summaries_steps is not None')
    
    if save_checkpoint_secs:
      raise ValueError(
          'logdir cannot be None when save_checkpoint_secs is not None')
    
  restored_graph = tf.Graph()
  with restored_graph.as_default():
    # Restore the metagraph.
    path_to_model = tf.train.latest_checkpoint(
        FLAGS.train_dir,
        latest_filename='checkpoint')
    if path_to_model is None:
      raise ValueError('No model checkpoint found in {}!'.format(FLAGS.train_dir))
    new_saver = tf.train.import_meta_graph(path_to_model + '.meta')
    print("Finished importing meta graph!")
    
    # Add a Hook to stop after specified number of steps.
    stop_after_n = th.TrustingStopAtStepHook(
        num_steps=FLAGS.max_number_of_steps)

    # Add a Hook to enable profiling.
    if FLAGS.profiling_enabled:
      profile_dir = FLAGS.profile_dir or FLAGS.train_dir
      profile_every_n = th.TrustingProfilerHook(  # tf.python.train.
          save_steps=FLAGS.profile_every_n_steps,
          save_secs=None,
          output_dir=profile_dir,
          show_dataflow=True,
          show_memory=False)
      hooks = [stop_after_n, profile_every_n]
    else:
      hooks = [stop_after_n]
  
    with ts.TrustingMonitoredTrainingSession(
        master='',
        is_chief=True,  # We're always chief, deploying on one GPU only.
        checkpoint_dir=logdir,
        scaffold=None,  # do finalize.
        hooks=hooks,
        chief_only_hooks=None,
        save_checkpoint_secs=save_checkpoint_secs,
        save_checkpoint_steps=save_checkpoint_steps,
        save_summaries_secs=save_summaries_secs,
        save_summaries_steps=save_summaries_steps,
        config=None,
        max_wait_secs=max_wait_secs) as session:
      # Restore values from checkpoint.
      new_saver.restore(session, path_to_model)
      
      # Find the train_op. It is the only op we added to slim_train_op.
      train_op = tf.get_collection('slim_train_op')[0]
      
      # Kick off training (like in train() in tf_slim_training.py).
      loss = None
      while not session.should_stop():
        loss = session.run(train_op, run_metadata=None)
  return loss


def main(_):
  sum_steps = None if FLAGS.save_summaries_secs is not None else FLAGS.save_summaries_steps
  cpt_steps = None if FLAGS.save_interval_secs is not None else FLAGS.save_checkpoints_steps
  sum_secs = None if FLAGS.save_summaries_steps is not None else FLAGS.save_summaries_secs
  cpt_secs = None if FLAGS.save_checkpoints_steps is not None else FLAGS.save_interval_secs
  
  # Restore and continue training.
  train(logdir=FLAGS.train_dir,
        save_summaries_steps=sum_steps,
        save_checkpoint_steps=cpt_steps,
        save_summaries_secs=sum_secs,
        save_checkpoint_secs=cpt_secs)


if __name__ == '__main__':
  app.run()
