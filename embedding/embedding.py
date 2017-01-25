import tensorflow as tf
from model.query2vec import Query2vec
from helper import model_helper
from config.config import FLAGS
from utils import data_utils
import os.path


class Embedding:
  def __init__(self):
    pass

  @staticmethod
  def query_embedding():
    q2v = Query2vec()
    # Create model and load parameters.
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options,
                              intra_op_parallelism_threads=20)) as sess:
      # Create model.
      print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
      with tf.device("/gpu:0"):
        model = model_helper.create_model(sess, True)
        model.batch_size = FLAGS.batch_size  # We decode one sentence at a time.

        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir,
                                  "vocab%d" % FLAGS.vocab_size)
        _, vocab_rev = data_utils.initialize_vocabulary(vocab_path)
        q2v.embedding_batch(sess, model, vocab_rev)
