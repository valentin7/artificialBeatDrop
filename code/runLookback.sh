bazel run //magenta/models/melody_rnn:melody_rnn_generate -- --config=lookback_rnn \
  --bundle_file=lookback_rnn.mag \
  --output_dir=../out/lookback_rnn/ \
  --num_outputs=3 \
  --num_steps=128 \
  --primer_midi=../midi/banger/sexy_bitch_synth.mid
