# loopextractor

A python script for extracting loops from audio files.

The script uses non-negative tensor factorization to model a version of the spectrum.

The code was written by Jordan B. L. Smith (@jblsmith) in December 2019.

### Usage

Calling loopextractor from the command line will run it on the included audio file as an example:

```
python loopextractor.py
```

You can also import it and use the functions in it on your own data:

```
import loopextractor
loopextractor.run_algorithm("my_audio_file.mp3", n_templates=[30,25,10], output_savename="my_string")
```

### Reference

The script implements the algorithm described in a paper I published in 2018, [described here](http://jblsmith.github.io/projects/nonnegative-tensor-factorization/). When using this code for an academic paper/project, please cite this paper as a reference:

> Smith, Jordan B. L., and Goto, Masataka. 2018. "Nonnegative tensor factorization for source separation of loops in audio." *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (IEEE ICASSP 2018).* Calgary, AB, Canada. pp. 171--175. 

The included example song was assembled using loops from [FreeSound.org](FreeSound.org) that were licensed Creative-Commons 0, i.e., committed to the public domain.

### License

This project is licensed under the terms of the [GNU Lesser General Public License version 3 (LGPLv3)](https://www.gnu.org/licenses/lgpl-3.0.en.html).

### Disclaimer

Although the code for loopextractor follows the same steps described in the ICASSP paper cited above,
this code was written from scratch in December 2019 by Jordan Smith alone.

Outside of this, the code for loopextractor has no relationship or connection to work done at AIST,
nor to the code that powers the Unmixer website (https://unmixer.ongaaccel.jp/).
