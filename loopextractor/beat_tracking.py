
def get_downbeats_with_librosa(signal, fs):
    """Use Librosa to estimate downbeats in a naive way.
    A 4/4 time signature is assumed and the phase with the best onset strength
    is used to set the downbeat.
    Parameters
    -------
    signal : np.ndarray [shape=(n,), dtype=float]
        Input mono audio signal.
    fs : int
        Sampling rate.
    Returns
    -------
    downbeat_times : np.ndarray [shape=(n,), dtype=float]
        List of estimated downbeat times in seconds.
    """
    import numpy as np
    import librosa
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=fs, units="frames")
    onset_strength_frames = librosa.onset.onset_strength(y=signal, sr=fs)
    phase_strengths = [np.median(onset_strength_frames[beat_frames[i::4]]) for i in range(4)]
    best_phase = np.argmax(phase_strengths)
    return librosa.frames_to_time(beat_frames[best_phase::4], sr=fs)


def get_downbeats_with_madmom(signal):
    """Use madmom package to estimate downbeats for an audio signal.
    Parameters
    ----------
    signal : np.ndarray [shape=(n,), dtype=float]
        Input mono audio signal.
    Returns
    -------
    downbeat_times : np.ndarray [shape=(n,), dtype=float]
        List of estimated downbeat times in seconds.
    Examples
    --------
    >>> signal_mono, fs = librosa.load("example_song.mp3", sr=None, mono=True)
    >>> get_downbeats_with_madmom(signal_mono)
    array([1.000e-02, 1.890e+00, 3.760e+00, 5.630e+00, 7.510e+00, 9.380e+00,
           1.126e+01, 1.313e+01, 1.501e+01, 1.688e+01, 1.876e+01, 2.064e+01,
           2.251e+01, 2.439e+01, 2.626e+01, 2.814e+01, 3.002e+01, 3.189e+01,
           3.376e+01, 3.564e+01, 3.751e+01, 3.939e+01, 4.126e+01, 4.314e+01,
           4.501e+01, 4.689e+01, 4.876e+01, 5.063e+01, 5.251e+01, 5.439e+01,
           5.626e+01, 5.813e+01])

    See Also
    --------
    madmom.features.downbeats.RNNDownBeatProcessor
    madmom.features.downbeats.DBNDownBeatTrackingProcessor
    """
    import madmom
    act = madmom.features.downbeats.RNNDownBeatProcessor()(signal)
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    processor_output = proc(act)
    downbeat_times = processor_output[processor_output[:, 1] == 1, 0]
    return downbeat_times