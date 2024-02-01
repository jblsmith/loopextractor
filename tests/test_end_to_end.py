import os

from loopextractor.loopextractor import run_algorithm

def test_run_algorithm():
    os.makedirs("tests/test_loops", exist_ok=True)
    run_algorithm("tests/example_song.mp3", n_templates=[0,0,0], output_savename="tests/test_loops/loop")
    for i in range(18):
        assert os.path.isfile(f"tests/test_loops/loop_{i}.wav")