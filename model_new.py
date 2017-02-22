# Working with TF commit 24466c2e6d32621cd85f0a78d47df6eed2c5c5a6
#coding=utf-8
import math

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.layers.python.layers import embedding_lookup_unique
from tensorflow.python.ops import embedding_ops
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
import time
import helpers
from seq2seq import decoder_fn
from seq2seq import seq2seq
from seq2seq.loss import sequence_loss
from tensorflow.python.framework import ops
# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "0.0.0.0:2221",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "0.0.0.0:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "worker", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
tf.app.flags.DEFINE_string("gpu", None, "specify the gpu to use")
tf.app.flags.DEFINE_integer("batchsize", 128, "batch_sizeIndex of task within the job")

class Seq2SeqModel():
    """Seq2Seq model usign blocks from new `tf.contrib.seq2seq`.
    Requires TF 1.0.0-alpha"""

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size,
                 bidirectional=True,
                 attention=False,
                 debug=False):
        self.debug = debug
        self.bidirectional = bidirectional
        self.attention = attention

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self._make_graph()

    @property
    def decoder_hidden_units(self):
        # @TODO: is this correct for LSTMStateTuple?
        return self.decoder_cell.output_size

    def _make_graph(self):
        if self.debug:
            self._init_debug_inputs()
        else:
            self._init_placeholders()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_debug_inputs(self):
        """ Everything is time-major """
        x = [[5, 6, 7],
             [7, 6, 0],
             [0, 7, 0]]
        xl = [2, 3, 1]
        self.encoder_inputs = tf.constant(x, dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.constant(xl, dtype=tf.int32, name='encoder_inputs_length')

        self.decoder_targets = tf.constant(x, dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.constant(xl, dtype=tf.int32, name='decoder_targets_length')

    def _init_placeholders(self):
        """ Everything is time-major """
        self.encoder_inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs',
        )
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        # required for training, not required for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets'
        )
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def _init_decoder_train_connectors(self):
        """
        During training, `decoder_targets`
        and decoder logits. This means that their shapes should be compatible.

        Here we do a bit of plumbing to set this up.
        """
        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1,
                                                        decoder_train_targets_seq_len,
                                                        on_value=self.EOS, off_value=self.PAD,
                                                        dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:

            # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            with ops.device("/cpu:0"):
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=initializer,
                    dtype=tf.float32)

                self.encoder_inputs_embedded = embedding_ops.embedding_lookup(
                    self.embedding_matrix, self.encoder_inputs)

                self.decoder_train_inputs_embedded = embedding_ops.embedding_lookup(
                    self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(cell=self.encoder_cell,
                                  inputs=self.encoder_inputs_embedded,
                                  sequence_length=self.encoder_inputs_length,
                                  parallel_iterations=16,
                                  time_major=True,
                                  dtype=tf.float32)
                )
    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:

            ((encoder_fw_outputs,
              encoder_bw_outputs),
             (encoder_fw_state,
              encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell,
                                                cell_bw=self.encoder_cell,
                                                inputs=self.encoder_inputs_embedded,
                                                sequence_length=self.encoder_inputs_length,
                                                time_major=True,
                                                dtype=tf.float32)
                )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_fw_outputs), 2)

            if isinstance(encoder_fw_state, LSTMStateTuple):

                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

            st = time.time()
            decoder_fn_train = decoder_fn.simple_decoder_fn_train(encoder_state=self.encoder_state)

            print("decoder_fn_inference:"+str(time.time()-st))

            (self.decoder_outputs_train,
             self.decoder_state_train,
             self.decoder_context_state_train) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.decoder_cell,
                    decoder_fn=decoder_fn_train,
                    inputs=self.decoder_train_inputs_embedded,
                    sequence_length=self.decoder_train_length,
                    time_major=True,
                    scope=scope,
                )
            )
            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = sequence_loss(logits=logits, targets=targets,
                                          weights=self.loss_weights)
        self.train_op = tf.train.AdadeltaOptimizer().minimize(self.loss)

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        targets_, targets_length_ = helpers.batch(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_inference_inputs(self, input_seq):
        inputs_, inputs_length_ = helpers.batch(input_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }


def make_seq2seq_model(**kwargs):
    args = dict(encoder_cell=GRUCell(786),
                decoder_cell=GRUCell(786),
                vocab_size=40000,
                embedding_size=150,
                attention=True,
                bidirectional=True,
                debug=False)
    args.update(kwargs)
    return Seq2SeqModel(**args)


def train_on_copy_task(session, model,
                       batch_size=128,
                       max_batches=5000,
                       batches_in_epoch=20,
                       verbose=True):

    loss_track = []
    try:
        import time
        for batch in range(max_batches+1):
            #batch_data = next(batches)
            encoder_inputs, decoder_inputs = helpers.train_data_sequences(batch_size)
            fd = model.make_train_inputs(encoder_inputs, decoder_inputs)
            start_time = time.time()
            _, l = session.run([model.train_op, model.loss], fd)
            print (time.time() - start_time)
            loss_track.append(l)


            if verbose:
                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(session.run(model.loss, fd)))
                    for i, (e_in, dt_pred) in enumerate(zip(
                            fd[model.encoder_inputs].T,
                            session.run(model.decoder_prediction_train, fd).T
                        )):
                        print('  sample {}:'.format(i + 1))
                        print('    enc input           > {}'.format(e_in))
                        print('    dec train predicted > {}'.format(dt_pred))
                        if i >= 2:
                            break
                    print()
    except KeyboardInterrupt:
        print('training interrupted')

    return loss_track

def create_model(checkpoint_dir, gpu="", max_batches=500000):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    print(ps_hosts)
    print(worker_hosts)
    print(FLAGS.job_name)
    print(FLAGS.task_index)
    task_index = FLAGS.task_index
    job_name = FLAGS.job_name
    batch_size = FLAGS.batchsize
    print(batch_size)
    if(job_name == "single"):
        master = ""
    else:
        cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=task_index)
        master = server.target
    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    else:
        # Device setting
        core_str = "cpu:0" if (gpu is None or gpu == "") else "gpu:%d" % int(gpu)
        if job_name == "worker":
            device = tf.train.replica_device_setter(cluster=cluster,
                                    worker_device='job:worker/task:%d/%s' % (task_index, core_str),
                                    ps_device='job:ps/task:%d/%s' % (task_index, core_str))
        else:
            device = "/" + core_str
        with tf.device(device):
            model = make_seq2seq_model(attention=False, bidirectional=False)

            init_op = tf.initialize_all_variables()
            saver = tf.train.Saver()
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=checkpoint_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=model.global_step,
                                 save_model_secs=60)
        gpu_options = tf.GPUOptions(allow_growth=True)
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        # log_device_placement=True,
                                        gpu_options=gpu_options,
                                        intra_op_parallelism_threads=16)
        with sv.prepare_or_wait_for_session(master=master, config=session_config) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [model.opt.chief_queue_runner])
                sess.run(model.opt.init_token_op)
            for batch in range(max_batches + 1):
                # batch_data = next(batches)
                encoder_inputs, decoder_inputs = helpers.train_data_sequences(batch_size)
                fd = model.make_train_inputs(encoder_inputs, decoder_inputs)
                start_time = time.time()
                _, l = sess.run([model.train_op, model.loss], fd)
                print (time.time() - start_time)


        sv.stop()
if __name__ == '__main__':
    gpu = FLAGS.gpu
    task_index = FLAGS.task_index
    create_model('./checkpoint', gpu)
