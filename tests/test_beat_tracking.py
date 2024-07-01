
from loopextractor.beat_tracking import get_downbeats_with_librosa, get_downbeats_with_madmom

import librosa

test_signal_mono, fs = librosa.load("tests/example_song.mp3", sr=None, mono=True)


def test_get_downbeats_with_librosa():
    lib_beats = get_downbeats_with_librosa(test_signal_mono, fs)
    assert len(lib_beats) == 32


def test_get_downbeats_with_madmom():
    mad_beats = get_downbeats_with_madmom(test_signal_mono)
    assert len(mad_beats) == 32
