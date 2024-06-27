'''
    File name: loopextractor.py
    Author: Jordan B. L. Smith
    Date created: 2 December 2019
    Date last modified: 1 February 2024
    License: GNU Lesser General Public License v3 (LGPLv3)
    Python Version: 3.8
'''

import argparse
import copy
import librosa
import numpy as np
import os
import soundfile
import tensorly
import tensorly.decomposition as tld
from sklearn.decomposition import NMF

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
    downbeat_times = get_downbeats(signal_mono, fs)
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
        soundfile.write("{0}_{1}.wav".format(output_savename,ith_loop), ith_loop_signal, fs)

def get_downbeats(signal, fs):
    """
    Basic, sloppy downbeat detection: use Librosa-tracked beats, assume 4/4,
    and use the phase with the best onset strength.
    """
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=fs, units="frames")
    onset_strength_frames = librosa.onset.onset_strength(y=signal, sr=fs)
    phase_strengths = [np.median(onset_strength_frames[beat_frames[i::4]]) for i in range(4)]
    best_phase = np.argmax(phase_strengths)
    return librosa.frames_to_time(beat_frames[best_phase::4], sr=fs)

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
    tensor : np.ndarray [shape=(n1,n2,n3), dtype=complex64]
        tensor containing spectrum slices

    Examples
    --------
    >>> signal_mono, fs = librosa.load("example_song.mp3", sr=None, mono=True)
    >>> downbeat_times = get_downbeats(signal_mono)
    >>> downbeat_frames = librosa.time_to_samples(downbeat_times, sr=fs)
    >>> spectral_cube = make_spectral_cube(signal_mono, downbeat_frames)
    >>> spectral_cube.shape
    (1025, 162, 31)
    >>> spectral_cube[:2,:2,:2]
    array([[[ 18.08905602+0.00000000e+00j, -20.48682976+0.00000000e+00j],
            [-16.07670403+0.00000000e+00j, -44.98669434+0.00000000e+00j]],

           [[-19.45080566+3.66026653e-15j,  -8.5700922 +3.14418630e-16j],
            [  1.01680577-3.67251587e+01j,  35.03190231-2.13507919e+01j]]])
    """
    assert len(signal_mono.shape) == 1
    # For each span of audio, compute the FFT using librosa defaults.
    usable_downbeat_frames = [d for d in downbeat_frames if d <= len(signal_mono)]
    fft_per_span = [librosa.core.stft(signal_mono[b1:b2]) for b1,b2 in zip(usable_downbeat_frames[:-1], usable_downbeat_frames[1:])]
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
    
    If ANY of the values for n_templates are invalid, than
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
    >>> validate_template_sizes(np.zeros((1025, 162, 31)), [100, 50, 20])
    array([100, 50, 20])
    >>> validate_template_sizes(np.zeros((1025, 162, 31)), [0, 0, 0])
    array([63, 21, 7])
    >>> validate_template_sizes(np.zeros((1025, 162, 31)), [100, 50, 40])
    array([63, 21, 7])
    
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

def purify_core_tensor(core, factors, new_rank, dim_to_reduce=2):
    """Reduce the size of the core tensor by modelling repeated content
    across loop recipes. The output is a more "pure" set of loop
    recipes that should be more distinct from each other.

    Parameters
    ----------
    core : np.ndarray [shape=(n1,n2,n3)]
        Core tensor to be compressed.
    factors : list [shape=(3,), dtype=np.ndarray]
        List of estimated templates
    new_rank : int
        The new size for the core tensor
    dim_to_reduce : int
        The dimension along which to compress the core tensor.
        (Default value 2 will reduce the number of loop types.)

    Returns
    -------
    new_core : np.ndarray [shape=(n1,n2,new_rank)]
        Compressed version of the core tensor
    new_factors : list [shape=(3,), dtype=np.ndarray]
        New list of templates.
        Note: two templates will be the same as before;
            only the template for the compressed dimension
            will be different.
    """
    assert new_rank < core.shape[dim_to_reduce]
    X = tensorly.unfold(core,dim_to_reduce)
    model = NMF(n_components=new_rank, init='nndsvd', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    # Re-construct core tensor and factors based on NMF factors from core tensor:
    new_shape = list(core.shape)
    new_shape[dim_to_reduce] = new_rank
    new_core = tensorly.fold(H, dim_to_reduce, new_shape)
    new_factors = copy.copy(factors)
    new_factors[dim_to_reduce] = np.dot(factors[dim_to_reduce],W)
    return new_core, new_factors

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
    array([1, 36, 577])
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
    >>> np.random.seed(0)
    >>> factors = [np.abs(np.random.randn(1025, 63)),
            np.abs(np.random.randn(162, 21)),
            np.abs(np.random.randn(31, 7))]
    >>> core = np.abs(np.random.randn(63,21,7))
    >>> create_loop_spectrum(factors[0], factors[1], core[:,:,0])
    array([[727.4153606 , 728.64591236, 625.76726056, ..., 512.94167141,
            592.2098947 , 607.10457107],
           [782.11991843, 778.09690543, 682.71895323, ..., 550.43525375,
            636.51448493, 666.35600624],
           [733.96209316, 720.17586837, 621.80762807, ..., 501.51192504,
            590.14018676, 605.44147057],
           ...,
           [772.43712078, 758.88473642, 654.35159419, ..., 522.69754588,
            628.84580165, 641.66347072],
           [677.58720601, 666.52484723, 583.92269705, ..., 471.24362278,
            558.17441475, 573.31864635],
           [768.96634561, 758.85553214, 639.21515256, ..., 525.83186141,
            634.04799161, 644.35772338]])
    """
    loop_spectrum = np.dot(np.dot(sounds, core_slice), rhythms.transpose())
    return loop_spectrum

def choose_bar_to_reconstruct(loop_templates, ith_loop):
    """...Choose... bar... to... reconstruct!
    
    For now, it just chooses the bar with the largest activation.
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
    >>> np.random.seed(0)
    >>> factors = [np.abs(np.random.randn(1025, 63)),
            np.abs(np.random.randn(162, 21)),
            np.abs(np.random.randn(31, 7))]
    >>> choose_bar_to_reconstruct(factors[2], 0)
    10
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
    >>> np.random.seed(0)
    >>> random_matrix = np.random.randn(1025,130)
    >>> loop_spectrum = np.abs(random_matrix) / np.max(random_matrix)
    >>> random_matrix_2 = np.random.randn(1025,130)
    >>> loop_spectrum_2 = np.abs(random_matrix_2) / np.max(random_matrix_2)
    >>> get_loop_signal(loop_spectrum, loop_spectrum_2)
    array([-5.7243928e-04, -2.3625907e-04, -3.8087784e-04, ...,
            9.2569360e-05,  3.9195133e-04, -2.4777438e-04], dtype=float32)
        
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

def write_all_loop_signals(core, factors, spectral_cube,
    fs=44100, output_savename="extracted_loop"):
    # Reconstruct each loop:
    n_loops = core.shape[2]
    for ith_loop in range(n_loops):
        # Multiply templates together to get real loop spectrum:
        loop_spectrum = create_loop_spectrum(factors[0], factors[1], core[:,:,ith_loop])
        # Choose best bar to reconstruct from (we will use its phase):
        bar_ind = choose_bar_to_reconstruct(factors[2], ith_loop)
        # Reconstruct loop signal by masking original spectrum:
        ith_loop_signal = get_loop_signal(loop_spectrum, spectral_cube[:,:,bar_ind])
        # Write signal to disk:
        soundfile.write("{0}_{1}.wav".format(output_savename,ith_loop), ith_loop_signal, fs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write this later")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("output_path", type=str, help="Prefix path for output files")
    args = parser.parse_args()
    run_algorithm(args.audio_file, [0,0,0], args.output_path)