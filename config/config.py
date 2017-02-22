import tensorflow as tf

# Run time variables


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
