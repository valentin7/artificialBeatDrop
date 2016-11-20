from mido import MidiFile
from collections import defaultdict
import numpy as np
import re
import tensorflow as tf
import time
import mido
from mido import Message, MidiFile, MidiTrack

def main():
    saveMidi()
    #generateBanger()

def generateBanger():
    mid = MidiFile('../midi/classical/deb_prel_format0.mid')

    output = mido.open_output()


    for message in mid.play():
        output.send(message)

    for i, track in enumerate(mid.tracks):
        print('Track {}: {}'.format(i, track.name))
        # for message in track:
        #     print(message)

def saveMidi():
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=12, time=0))
    track.append(Message('note_on', note=64, velocity=64, time=32))
    track.append(Message('note_off', note=64, velocity=127, time=32))

    mid.save('new_song.mid')

if __name__ == '__main__':
    main()
