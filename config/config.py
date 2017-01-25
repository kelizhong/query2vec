import tensorflow as tf

# Run time variables
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("embedding_size", 150, "Size of each model layer.")
tf.app.flags.DEFINE_integer("size", 786, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")

tf.app.flags.DEFINE_string("data_dir", "./data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data/", "Training directory.")
tf.app.flags.DEFINE_string("embedding_dir", "./embedding_data/", "Embedding directory.")

# tf.app.flags.DEFINE_integer("vocab_size", 63200, "Size of vocab")
tf.app.flags.DEFINE_integer("vocab_size", 40000, "Size of vocab")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_per_summary", 20,
                            "How many training steps to do per summary.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_string("seq2seq_model", "seq2seq_model.ckpt", "seq2seq model checkpoint")
tf.app.flags.DEFINE_string("embedding_model", "embedding_model.ckpt", "embedding model checkpoint")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(3, 10), (3, 20), (5, 20), (7, 30)]
# _buckets = [(5, 10), (5, 20)]

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

NAME_ID = 566  # Change this for your data set

name = 'q2v'  # Extra credit: Make it name itself using another RNN

_holders = {NAME_ID: name}
