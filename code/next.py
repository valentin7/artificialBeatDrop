def sequence_proto_to_midi_file(sequence, output_file):
  """Convert tensorflow.magenta.NoteSequence proto to a MIDI file on disk.
  Time is stored in the NoteSequence in absolute values (seconds) as opposed to
  relative values (MIDI ticks). When the NoteSequence is translated back to
  MIDI the absolute time is retained. The tempo map is also recreated.
  Args:
    sequence: A tensorfow.magenta.NoteSequence proto.
    output_file: String path to MIDI file that will be written.
  """
  pretty_midi_object = sequence_proto_to_pretty_midi(sequence)
  pretty_midi_object.write(output_file)
