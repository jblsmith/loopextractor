# loop-extractor

A python scipt for extracting loops from audio files.

The script uses non-negative tensor factorization to model a version of the spectrum.

### Usage

You can call it on the reference example file from the command line:

```
python loop-extractor.py
```

You can also import it and use the functions in it on your own data:

```
import loop_extractor
loop_extractor.run_algorithm("my_audio_file.mp3", n_templates=[30,25,10], output_savename="my_string")
```

### Reference

The script implements the algorithm described in this paper:

    Smith, Jordan B. L., and Goto, Masataka. 2018. "Nonnegative tensor factorization for source separation of loops in audio." *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing.* Calgary, AB, Canada. 171–5.

The included example song was assembled using loops from [FreeSound.org](FreeSound.org) that were licensed Creative-Commons 0, i.e., committed to the public domain.

### License

This project is licensed under the terms of the MIT license.