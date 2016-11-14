from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
#from tqdm import tqdm
import tensorflow as tf
import numpy as np
import re
import operator
import math


def main():



def build_graph(mode, config, sequence_protos=None):
  """Builds the TensorFlow graph.
  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: A MelodyRnnConfig containing the MelodyEncoderDecoder and HParams to
        use.
    sequence_protos: A string path to a TFRecord file containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation.
  Returns:
    A tf.Graph instance which contains the TF ops.
  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate', or if
        sequence_protos does not match a file when mode is 'train' or
        'eval'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label
    #with tf.Graph().as_default() as graph:


  inputs, labels, lengths, = None, None, None
  state_is_tuple = True

  if mode == 'train' or mode == 'eval':
    inputs, labels, lengths = get_padded_batch(
        [sequence_protos], hparams.batch_size, input_size)

  elif mode == 'generate':
    inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                         input_size])
    # If state_is_tuple is True, the output RNN cell state will be a tuple
    # instead of a tensor. During training and evaluation this improves
    # performance. However, during generation, the RNN cell state is fed
    # back into the graph with a feed dict. Feed dicts require passed in
    # values to be tensors and not tuples, so state_is_tuple is set to False.
    state_is_tuple = False

  cells = []
  for num_units in hparams.rnn_layer_sizes:
    cell = tf.nn.rnn_cell.BasicLSTMCell(
        num_units, state_is_tuple=state_is_tuple)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=hparams.dropout_keep_prob)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if hparams.attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, hparams.attn_length, state_is_tuple=state_is_tuple)
  initial_state = cell.zero_state(hparams.batch_size, tf.float32)

  outputs, final_state = tf.nn.dynamic_rnn(
      cell, inputs, lengths, initial_state, parallel_iterations=1,
      swap_memory=True)

  outputs_flat = tf.reshape(outputs, [-1, hparams.rnn_layer_sizes[-1]])
  logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

  if mode == 'train' or mode == 'eval':
    if hparams.skip_first_n_losses:
      logits = tf.reshape(logits_flat, [hparams.batch_size, -1, num_classes])
      logits = logits[:, hparams.skip_first_n_losses:, :]
      logits_flat = tf.reshape(logits, [-1, num_classes])
      labels = labels[:, hparams.skip_first_n_losses:]

    labels_flat = tf.reshape(labels, [-1])
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits_flat, labels_flat)
    loss = tf.reduce_mean(softmax_cross_entropy)
    perplexity = tf.reduce_mean(tf.exp(softmax_cross_entropy))

    correct_predictions = tf.to_float(
        tf.nn.in_top_k(logits_flat, labels_flat, 1))
    accuracy = tf.reduce_mean(correct_predictions) * 100

    event_positions = tf.to_float(tf.not_equal(labels_flat, no_event_label))
    event_accuracy = tf.truediv(
        tf.reduce_sum(tf.mul(correct_predictions, event_positions)),
        tf.reduce_sum(event_positions)) * 100

    no_event_positions = tf.to_float(tf.equal(labels_flat, no_event_label))
    no_event_accuracy = tf.truediv(
        tf.reduce_sum(tf.mul(correct_predictions, no_event_positions)),
        tf.reduce_sum(no_event_positions)) * 100

    global_step = tf.Variable(0, trainable=False, name='global_step')

    tf.add_to_collection('loss', loss)
    tf.add_to_collection('perplexity', perplexity)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('global_step', global_step)

    summaries = [
        tf.scalar_summary('loss', loss),
        tf.scalar_summary('perplexity', perplexity),
        tf.scalar_summary('accuracy', accuracy),
        tf.scalar_summary('event_accuracy', event_accuracy),
        tf.scalar_summary('no_event_accuracy', no_event_accuracy),
    ]

    if mode == 'train':
      learning_rate = tf.train.exponential_decay(
          hparams.initial_learning_rate, global_step, hparams.decay_steps,
          hparams.decay_rate, staircase=True, name='learning_rate')

      opt = tf.train.AdamOptimizer(learning_rate)
      params = tf.trainable_variables()
      gradients = tf.gradients(loss, params)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                    hparams.clip_norm)
      train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                     global_step)
      tf.add_to_collection('learning_rate', learning_rate)
      tf.add_to_collection('train_op', train_op)

      summaries.append(tf.scalar_summary('learning_rate', learning_rate))

    if mode == 'eval':
      summary_op = tf.merge_summary(summaries)
      tf.add_to_collection('summary_op', summary_op)

  elif mode == 'generate':
    temperature = tf.placeholder(tf.float32, [])
    softmax_flat = tf.nn.softmax(
        tf.div(logits_flat, tf.fill([num_classes], temperature)))
    softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('initial_state', initial_state)
    tf.add_to_collection('final_state', final_state)
    tf.add_to_collection('temperature', temperature)
    tf.add_to_collection('softmax', softmax)





  return graph

def midi_files_to_sequence_proto(midi_files, batch_size, input_size):
    sequences = [midi_file_to_sequence_proto(midi_file, batch_size, input_size) for midi_file in midi_files]
    return get_padded_batch(sequences, batch_size, input_size)

def midi_file_to_sequence_proto(midi_file, batch_size, input_size):
  """Converts MIDI file to a tensorflow.magenta.NoteSequence proto.
  Args:
    midi_file: A string path to a MIDI file.
  Returns:
    A tensorflow.magenta.Sequence proto.
  Raises:
    MIDIConversionError: Invalid midi_file.
  """
  with tf.gfile.Open(midi_file, 'r') as f:
    midi_data = f.read()

  # In practice many MIDI files cannot be decoded with pretty_midi. Catch all
  # errors here and try to log a meaningful message. So many different
  # exceptions are raised in pretty_midi.PrettyMidi that it is cumbersome to
  # catch them all only for the purpose of error logging.
  # pylint: disable=bare-except
  if isinstance(midi_data, pretty_midi.PrettyMIDI):
    midi = midi_data
  else:
    try:
      midi = pretty_midi.PrettyMIDI(StringIO(midi_data))
    except:
      raise MIDIConversionError('Midi decoding error %s: %s' %
                                (sys.exc_info()[0], sys.exc_info()[1]))
  # pylint: enable=bare-except

  sequence = music_pb2.NoteSequence()

  # Populate header.
  sequence.ticks_per_quarter = midi.resolution
  sequence.source_info.parser = music_pb2.NoteSequence.SourceInfo.PRETTY_MIDI
  sequence.source_info.encoding_type = (
      music_pb2.NoteSequence.SourceInfo.MIDI)

  # Populate time signatures.
  for midi_time in midi.time_signature_changes:
    time_signature = sequence.time_signatures.add()
    time_signature.time = midi_time.time
    time_signature.numerator = midi_time.numerator
    try:
      # Denominator can be too large for int32.
      time_signature.denominator = midi_time.denominator
    except ValueError:
      raise MIDIConversionError('Invalid time signature denominator %d' %
                                midi_time.denominator)

  # Populate key signatures.
  for midi_key in midi.key_signature_changes:
    key_signature = sequence.key_signatures.add()
    key_signature.time = midi_key.time
    key_signature.key = midi_key.key_number % 12
    midi_mode = midi_key.key_number / 12
    if midi_mode == 0:
      key_signature.mode = key_signature.MAJOR
    elif midi_mode == 1:
      key_signature.mode = key_signature.MINOR
    else:
      raise MIDIConversionError('Invalid midi_mode %i' % midi_mode)

  # Populate tempo changes.
  tempo_times, tempo_qpms = midi.get_tempo_changes()
  for time_in_seconds, tempo_in_qpm in zip(tempo_times, tempo_qpms):
    tempo = sequence.tempos.add()
    tempo.time = time_in_seconds
    tempo.qpm = tempo_in_qpm

  # Populate notes by gathering them all from the midi's instruments.
  # Also set the sequence.total_time as the max end time in the notes.
  midi_notes = []
  midi_pitch_bends = []
  midi_control_changes = []
  for num_instrument, midi_instrument in enumerate(midi.instruments):
    for midi_note in midi_instrument.notes:
      if not sequence.total_time or midi_note.end > sequence.total_time:
        sequence.total_time = midi_note.end
      midi_notes.append((midi_instrument.program, num_instrument,
                         midi_instrument.is_drum, midi_note))
    for midi_pitch_bend in midi_instrument.pitch_bends:
      midi_pitch_bends.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_pitch_bend))
    for midi_control_change in midi_instrument.control_changes:
      midi_control_changes.append(
          (midi_instrument.program, num_instrument,
           midi_instrument.is_drum, midi_control_change))

  for program, instrument, is_drum, midi_note in midi_notes:
    note = sequence.notes.add()
    note.instrument = instrument
    note.program = program
    note.start_time = midi_note.start
    note.end_time = midi_note.end
    note.pitch = midi_note.pitch
    note.velocity = midi_note.velocity
    note.is_drum = is_drum

  for program, instrument, is_drum, midi_pitch_bend in midi_pitch_bends:
    pitch_bend = sequence.pitch_bends.add()
    pitch_bend.instrument = instrument
    pitch_bend.program = program
    pitch_bend.time = midi_pitch_bend.time
    pitch_bend.bend = midi_pitch_bend.pitch
    pitch_bend.is_drum = is_drum

  for program, instrument, is_drum, midi_control_change in midi_control_changes:
    control_change = sequence.control_changes.add()
    control_change.instrument = instrument
    control_change.program = program
    control_change.time = midi_control_change.time
    control_change.control_number = midi_control_change.number
    control_change.control_value = midi_control_change.value
    control_change.is_drum = is_drum

  return sequence


def make_sequence_example(inputs, labels):
  """Returns a SequenceExample for the given inputs and labels.
  Args:
    inputs: A list of input vectors. Each input vector is a list of floats.
    labels: A list of ints.
  Returns:
    A tf.train.SequenceExample containing inputs and labels.
  """
  input_features = [
      tf.train.Feature(float_list=tf.train.FloatList(value=input_))
      for input_ in inputs]
  label_features = [
      tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
      for label in labels]
  feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features),
      'labels': tf.train.FeatureList(feature=label_features)
  }
  feature_lists = tf.train.FeatureLists(feature_list=feature_list)
  return tf.train.SequenceExample(feature_lists=feature_lists)


def get_padded_batch(sequence, batch_size, input_size):
  """Reads batches of SequenceExamples from TFRecords and pads them.
  Can deal with variable length SequenceExamples by padding each batch to the
  length of the longest sequence with zeros.
  Args:
    file_list: A list of paths to TFRecord files containing SequenceExamples.
    batch_size: The number of SequenceExamples to include in each batch.
    input_size: The size of each input vector. The returned batch of inputs
        will have a shape [batch_size, num_steps, input_size].
    num_enqueuing_threads: The number of threads to use for enqueuing
        SequenceExamples.
  Returns:
    inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    labels: A tensor of shape [batch_size, num_steps] of int64s.
    lengths: A tensor of shape [batch_size] of int32s. The lengths of each
        SequenceExample before padding.
  """
  num_enqueuing_threads = 4
  # file_queue = tf.train.string_input_producer(file_list)
  # reader = tf.TFRecordReader()
  # _, serialized_example = reader.read(file_queue)


  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[input_size],
                                           dtype=tf.float32),
      'labels': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.int64)}

  #_, sequence = tf.parse_single_sequence_example(
    #  serialized_example, sequence_features=sequence_features)

  length = tf.shape(sequence['inputs'])[0]

  queue = tf.PaddingFIFOQueue(
      capacity=1000,
      dtypes=[tf.float32, tf.int64, tf.int32],
      shapes=[(None, input_size), (None,), ()])

  enqueue_ops = [queue.enqueue([sequence['inputs'],
                                sequence['labels'],
                                length])] * num_enqueuing_threads
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(batch_size)

if __name__ == '__main__':
    main()
