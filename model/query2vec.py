# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from utils import data_utils
from helper import model_helper
from config.config import FLAGS, _buckets

class Query2vec:
  def __init__(self):
    pass

  @staticmethod
  def train():
    """Train a query2vec model"""
    # Prepare train data.
    print("Preparing Seq2seq Model in %s" % FLAGS.train_dir)
    train_data, test_data, _ = data_utils.prepare_data(FLAGS.train_dir, FLAGS.vocab_size)
    checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.seq2seq_model)

    print("Loading training data from %s" % train_data)
    print("Loading development data from %s" % test_data)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options,
                                          intra_op_parallelism_threads=20)) as sess:
      # Create model.
      print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
      with tf.device("/gpu:0"):
        model = model_helper.create_model(sess, False)

      # Read data into buckets and compute their sizes.
      print("Reading development and training data (limit: %d)."
            % FLAGS.max_train_data_size)
      test_set = data_utils.read_data(test_data)
      train_set = data_utils.read_data(train_data, max_size=FLAGS.max_train_data_size)
      train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
      train_total_size = float(sum(train_bucket_sizes))

      # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
      # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
      # the size if i-th training bucket, as used later.
      train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                             for i in xrange(len(train_bucket_sizes))]

      # This is the training loop.
      step_time, loss = 0.0, 0.0
      current_step = 0
      previous_losses = []
      prev_loss = [1000000] * len(_buckets)

      train_writer = tf.summary.FileWriter(os.path.join("summary/train"), sess.graph)
      test_writer = tf.summary.FileWriter(os.path.join("summary/test"), sess.graph)
      while True:
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set[bucket_id], bucket_id)
        summaries, _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1
        if current_step % FLAGS.steps_per_summary == 0:
          train_writer.add_summary(summaries, current_step)
          train_writer.flush()
          print('Step: %s' % current_step)
        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          # Print statistics for the previous epoch.
          perplexity = math.exp(loss) if loss < 300 else float('inf')
          print("global step %d learning rate %.4f step-time %.2f perplexity "
                "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity))
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)
          # Save checkpoint and zero timer and loss.
          step_time, loss = 0.0, 0.0
          # Run evals on development set and print their perplexity.
          count = 0
          for bucket_id in xrange(len(_buckets)):
            if len(test_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              test_set[bucket_id], bucket_id)
            summaries, _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                    target_weights, bucket_id, True)
            test_writer.add_summary(summaries, current_step)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            if eval_ppx < prev_loss[bucket_id]:
              prev_loss[bucket_id] = eval_ppx
              count += 1
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

          if count > len(_buckets) / 3:
            print("saving model...")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          sys.stdout.flush()
          test_writer.flush()

  @staticmethod
  def embedding_batch(sess, model, vocab):
    """
    Fast responses by passing
    - pre-generated model,
    - session
    - vocabulary
    And a batch sentences to produce a output logit to
    """
    checkpoint_path = os.path.join(FLAGS.embedding_dir, FLAGS.embedding_model)
    writer = tf.summary.FileWriter(FLAGS.embedding_dir, sess.graph)
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'item_embedding'
    embed.metadata_path = data_utils.get_metadata_set_path(FLAGS.embedding_dir)
    projector.visualize_embeddings(writer, config)
    train_path = data_utils.get_train_set_path(FLAGS.train_dir)
    train_ids_path = train_path + ("_ids%d" % FLAGS.vocab_size)
    train_set = data_utils.read_data(train_ids_path, max_size=500000)  # FLAGS.max_train_data_size)
    state_list = []

    meta_list = []
    for bucket_id in xrange(len(_buckets)):
      meta_list.extend(train_set[bucket_id])
    for i, each in enumerate(meta_list):
      meta_list[i] = [vocab[id] for id in each[0]]
    deduped = {}
    for i, each in enumerate(meta_list):
      deduped[" ".join(each)] = i
    deduped_tuple_list = deduped.items()
    indices = [each[1] for each in deduped_tuple_list]

    metadata_path = embed.metadata_path

    with open(metadata_path, 'w+') as item_file:
      item_file.write('id\tchar\n')
      for i, each in enumerate(deduped_tuple_list):
        item_file.write('{}\t{}\n'.format(i, each[0]))
      print('metadata file created')

    for bucket_id in xrange(len(_buckets)):
      begin = 0
      # some data will be ignored
      while begin < len(train_set[bucket_id]):
        bucket_data = train_set[bucket_id]
        data = bucket_data[begin: begin + FLAGS.batch_size]
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data, bucket_id, False)
        states, last_states = model.step_encoder_decoder(sess, encoder_inputs, decoder_inputs,
                                                         target_weights, bucket_id, True)
        state_list.append(states)
        begin += FLAGS.batch_size

    concat = np.concatenate(state_list, axis=0)
    embedding_states = concat[indices]

    item_embedding = tf.get_variable(embed.tensor_name, [len(deduped_tuple_list), FLAGS.size])
    assign_op = item_embedding.assign(embedding_states)
    sess.run(assign_op)
    saver = tf.train.Saver([item_embedding])
    saver.save(sess, checkpoint_path, global_step=model.global_step)
