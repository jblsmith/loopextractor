'''
    File name: loopextractor.py
    Author: Jordan B. L. Smith
    Date created: 2 December 2019
    Date last modified: 18 December 2019
    License: GNU Lesser General Public License v3 (LGPLv3)
    Python Version: 3.7
'''

import librosa
import madmom
import numpy as np
import os
import tensorly
import tensorly.decomposition as tld

def run_algorithm(audio_file, n_templates=[0,0,0], output_savename="extracted_loop"):
    """Complete pipeline of algorithm.

    Parameters
    ----------
    audio_file : string
        Path to audio file to be loaded and analysed.
    n_templates : list of length 3
        The number of sound, rhythm and loop templates.
        Default value (0,0,0) causes the script to estimate reasonable values.
    output_savename: : string
        Base string for saved output filenames.

    Returns
    -------
    A set of files containing the extracted loops.

    Examples
    --------
    >>> run_algorithm("example_song.mp3", [40,20,7], "extracted_loop")
    
    See also
    --------
    tensorly.decomposition.non_negative_tucker
    """
    assert os.path.exists(audio_file)
    assert len(n_templates)==3
    assert type(n_templates) is list
    # Load mono audio:
    signal_mono, fs = librosa.load(audio_file, sr=None, mono=True)
    # Use madmom to estimate the downbeat times:
    downbeat_times = get_downbeats(signal_mono)
    # Convert times to frames so we segment signal:
    downbeat_frames = librosa.time_to_samples(downbeat_times, sr=fs)
    # Create spectral cube out of signal:
    spectral_cube = make_spectral_cube(signal_mono, downbeat_frames)
    # Validate the input n_templates (inventing new ones if any is wrong):
    n_sounds, n_rhythms, n_loops = validate_template_sizes(spectral_cube, n_templates)
    # Use TensorLy to do the non-negative Tucker decomposition:
    core, factors = tld.non_negative_tucker(np.abs(spectral_cube), [n_sounds, n_rhythms, n_loops], n_iter_max=500, verbose=True)
    # Reconstruct each loop:
    for ith_loop in range(n_loops):
        # Multiply templates together to get real loop spectrum:
        loop_spectrum = create_loop_spectrum(factors[0], factors[1], core[:,:,ith_loop])
        # Choose best bar to reconstruct from (we will use its phase):
        bar_ind = choose_bar_to_reconstruct(factors[2], ith_loop)
        # Reconstruct loop signal by masking original spectrum:
        ith_loop_signal = get_loop_signal(loop_spectrum, spectral_cube[:,:,bar_ind])
        # Write signal to disk:
        librosa.output.write_wav("{0}_{1}.wav".format(output_savename,ith_loop), ith_loop_signal, fs)

def get_downbeats(signal):
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
    >>> get_downbeats(signal_mono)
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
    act = madmom.features.downbeats.RNNDownBeatProcessor()(signal)
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    processor_output = proc(act)
    downbeat_times = processor_output[processor_output[:,1]==1,0]
    return downbeat_times

def make_spectral_cube(signal_mono, downbeat_frames):
    """Convert audio signal into a spectral cube using
    specified downbeat frames.

    An STFT is taken of each segment of audio, and
    these STFTs are stacked into a 3rd dimension.
    
    The STFTs may have different lengths; they are
    zero-padded to the length of the longest STFT.

    Parameters
    ----------
    signal_mono : np.ndarray [shape=(n,), dtype=float]
        one-dimensional audio signal to convert
    downbeat_frames : np.ndarray [shape=(n,), dtype=int]
        list of frames separating downbeats (or whatever
        time interval is desired)

    Returns
    -------
    tensor  : np.ndarray [shape=(n1,n2,n3), dtype=complex64]
        tensor containing spectrum slices

    Examples
    --------
    >>> spectral_cube = make_spectral_cube(signal_mono, downbeat_frames)
    >>> spectral_cube[:2,:2,:2]
    array([[[ 18.08905602+0.00000000e+00j, -20.48682976+0.00000000e+00j],
            [-16.07670403+0.00000000e+00j, -44.98669434+0.00000000e+00j]],

           [[-19.45080566+3.66026653e-15j,  -8.5700922 +3.14418630e-16j],
            [  1.01680577-3.67251587e+01j,  35.03190231-2.13507919e+01j]]])
    """
    assert len(signal_mono.shape) == 1
    # For each span of audio, compute the FFT using librosa defaults.
    fft_per_span = [librosa.core.stft(signal_mono[b1:b2]) for b1,b2 in zip(downbeat_frames[:-1],downbeat_frames[1:])]
    # Tensor size 1: the number of frequency bins
    freq_bins = fft_per_span[0].shape[0]
    # Tensor size 2: the length of the STFTs.
    # This could vary for each span; use the maximum.
    rhyt_bins = np.max([fpb.shape[1] for fpb in fft_per_span])
    # Tensor size 3: the number of spans.
    bar_bins = len(fft_per_span)
    tensor = np.zeros((freq_bins, rhyt_bins, bar_bins)).astype(complex)
    for i in range(bar_bins):
        tensor[:,:fft_per_span[i].shape[1],i] = fft_per_span[i]
    return tensor

def validate_template_sizes(spectral_cube, n_templates):
    """Ensure that specified number of estimated templates are valid.
    Values must be greater than 1 and strictly less than
    the corresponding dimension of the original tensor.
    So, if the tensor has size [1025,100,20], then
    n_templates = [99,99,10] is valid (though unadvised), while
    n_templates = [30,20,20] is invalid.
    
    If any of the values for n_templates are invalid, than
    get_recommended_template_sizes() is used to obtain
    replacement values for n_templates.

    Parameters
    ----------
    spectral_cube : np.ndarray [shape=(n1,n2,n3)]
        Original tensor to be modeled.
    n_templates : list [shape=(3,), dtype=int]
        Proposed numbers of templates.

    Returns
    -------
    output_n_templates : np.ndarray [shape=(3,), dtype=int]
        Validated numbers of templates.

    Examples
    --------
    >>> validate_template_sizes(spectral_cube, n_templates)
    array([63, 21,  7])
    
    See Also
    --------
    get_recommended_template_sizes
    """
    max_template_sizes = np.array(spectral_cube.shape) - 1
    min_template_sizes = np.ones_like(max_template_sizes)
    big_enough = np.all(min_template_sizes <= n_templates)
    small_enough = np.all(n_templates <= max_template_sizes)
    valid = big_enough & small_enough
    if valid:
        return n_templates
    else:
        return get_recommended_template_sizes(spectral_cube)

def get_recommended_template_sizes(spectral_cube):
    """Propose reasonable values for numbers of templates
    to estimate.
    
    If a dimension of the tensor is N, then N^(6/10), rounded
    down, seems to give a reasonable value.

    Parameters
    ----------
    spectral_cube : np.ndarray [shape=(n1,n2,n3)]
        Original tensor to be modeled.

    Returns
    -------
    recommended_sizes : np.ndarray [shape=(len(spectral_cube.shape),), dtype=float]
        Suggested number of templates.

    Examples
    --------
    >>> get_recommended_template_sizes(np.zeros((100,200,300)))
    array([15, 23, 30])
    >>> get_recommended_template_sizes(np.zeros((4,400,40000)))
    array([  1,  36, 577])
    """
    max_template_sizes = np.array(spectral_cube.shape) - 1
    min_template_sizes = np.ones_like(max_template_sizes)
    recommended_sizes = np.floor(max_template_sizes**.6).astype(int)
    recommended_sizes = np.max((recommended_sizes, min_template_sizes),axis=0)
    assert np.all(min_template_sizes <= recommended_sizes)
    assert np.all(recommended_sizes <= max_template_sizes)
    return recommended_sizes

def create_loop_spectrum(sounds, rhythms, core_slice):
    """Recreate loop spectrum from a slice of the core tensor
    and the first two templates, the sounds and rhythms.

    Parameters
    ----------
    sounds : np.ndarray [shape=(n_frequency_bins, n_sounds), dtype=float]
        The sound templates, one spectral template per column.
    rhythms : np.ndarray [shape=(n_time_bins, n_rhythms), dtype=float]
        The rhythm templates, or time-in-bar activations functions.
        One rhythm template per column.
    core_slice : np.ndarray [shape=(n_sounds, n_rhythms)]
        A slice of the core tensor giving the recipe for one loop.

    Returns
    -------
    loop_spectrum : np.ndarray [shape=(n_frequency_bins, n_time_bins), dtype=float]
        Reconstruction of spectrum.

    Examples
    --------
    >>> create_loop_spectrum(factors[0], factors[1], core[:,:,0])
    array([[2.18655152e+01, 2.33809279e+01, 1.41177489e+01, ...,
            3.41226316e+00, 4.13603762e+00, 4.40903069e+00],
           [2.19071941e+01, 1.72018802e+01, 9.34874138e+00, ...,
            1.03270183e+01, 1.24937576e+01, 1.55722056e+01],
           [2.41047662e+01, 2.29424785e+01, 2.85487624e+01, ...,
            1.20072163e+01, 2.41067461e+01, 2.76246296e+01],
           ...,
           [6.43656317e-02, 4.73108798e-02, 1.82280916e-02, ...,
            3.22615535e-02, 1.46806524e-02, 1.14091573e-02],
           [4.54570758e-02, 3.00483403e-02, 1.52157357e-02, ...,
            3.03952309e-02, 1.39044458e-02, 1.07467596e-02],
           [1.62624878e-01, 1.40923815e-01, 4.41798227e-02, ...,
            2.62436762e-02, 1.22221164e-02, 1.24432306e-02]])
    """
    loop_spectrum = np.dot(np.dot(sounds, core_slice), rhythms.transpose())
    return loop_spectrum

def choose_bar_to_reconstruct(loop_templates, ith_loop):
    """...Choose... bar... to... reconstruct!
    
    For now, it just choose the bar with the largest activation.
    More information could / should be included, like reducing
    cross-talk, which would mean considering the activations (but
    ideally the relative *loudnesses*) of the other loops.

    Parameters
    ----------
    loop_templates : np.ndarray [shape=(n_bars, n_loop_types), dtype=float]
        The loop activation templates, one template per column.
    ith_loop : int
        The index of the loop template.

    Returns
    -------
    bar_ind : int
        The index of the bar to choose.

    Examples
    --------
    >>> choose_bar_to_reconstruct(factors[2], 0)
    30
    """
    bar_ind = np.argmax(loop_templates[:,ith_loop])
    return bar_ind

def get_loop_signal(loop_spectrum, original_spectrum):
    """Reconstruct the signal for a loop given its spectrum
    and the original spectrum.
    
    The original spectrum is used as the basis, and the reconstructed
    loop spectrum is used to mask the spectrum.

    Parameters
    ----------
    loop_spectrum : np.ndarray [shape=(n_freq_bins, n_time_bins_1), dtype=float]
        Reconstructed loop spectrum (real)
    original_spectrum : np.ndarray [shape=(n_freq_bins, n_time_bins_2), dtype=complex]
        Original spectrum (complex; possibly different length of time)

    Returns
    -------
    signal : np.ndarray [shape=(n,), dtype=float]
        Estimated signal of isolated loop.

    Examples
    --------
    >>> get_loop_signal(loop_spectrum, spectral_cube[:,:,30])
    get_loop_signal(loop_spectrum, spectral_cube[:,:,30])
    array([-0.08186008, -0.08577164, -0.06582578, ...,  0.00286376,
            0.00289357,  0.00292362], dtype=float32)
    
    See also
    --------
    librosa.util.softmask
    """
    assert loop_spectrum.shape[0] == original_spectrum.shape[0]
    min_length = np.min((loop_spectrum.shape[1], original_spectrum.shape[1]))
    orig_mag, orig_phase = librosa.magphase(original_spectrum)
    mask = librosa.util.softmask(loop_spectrum[:,:min_length], orig_mag[:,:min_length], power=1)
    masked_spectrum = original_spectrum[:,:min_length] * mask
    signal = librosa.core.istft(masked_spectrum)
    return signal

if __name__ == "__main__":
	# Run algorithm on test song:
	run_algorithm("loopextractor/audio/example_song.mp3", n_templates=[0,0,0], output_savename="extracted_loop")
