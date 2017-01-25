from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from config.config import FLAGS, _buckets
from external import seq2seq_model

def create_model(session, forward_only):
  """Create the query2vec model and initialize or load parameters into the session."""
  model = seq2seq_model.Seq2SeqModel(
    FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
    FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
    FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
    forward_only=forward_only)

  for each in tf.global_variables():
    print(each.name, each.device)

  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

  if ckpt and ckpt.model_checkpoint_path:
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model
