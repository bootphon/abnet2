import numpy as np
from glob import glob
import os.path as path
from spectral import Spectral
from scipy.io import wavfile


wav_dir = 'data/wav'
alignment_file = 'data/alignment.txt'

def do_fbank(fname):
    srate, sound = wavfile.read(fname)
    fbanks = Spectral(
        nfilt=40,               # nb of filters in mel bank
        alpha=0.97,             # pre-emphasis
        do_dct=False,           # we do not want MFCCs
        fs=srate,               # sampling rate
        frate=100,              # frame rate
        wlen=0.025,             # window length
        nfft=1024,              # length of dft
        do_deltas=False,        # speed
        do_deltasdeltas=False   # acceleration
    )
    fb = np.array(fbanks.transform(sound), dtype='float32')
    return fb

def read_alignment():
    labels = set()
    alignments = {}
    with open(alignment_file) as fin:
        for line in fin:
            splitted = line.split()
            alignment = [phn.split('_')[0] for phn in splitted[1:]]
            labels.update(alignment)
            alignments[splitted[0]] = alignment
    labels = {l: i for i, l in enumerate(labels)}
    return labels, alignments

def write_fbanks():
    wavs = glob(path.join(wav_dir, '*.wav'))
    labels, alignments = read_alignment()
    fbanks = {path.splitext(path.basename(wavfile))[0]: do_fbank(wavfile)
              for wavfile in wavs}
    X = {}
    for f in fbanks:
        nframes = len(alignments[f])
        X[f] = fbanks[f][1:nframes+1]
    np.savez('data/fbanks.npz', **X)

if __name__ == '__main__':
    write_fbanks()
