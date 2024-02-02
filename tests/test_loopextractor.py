from loopextractor.loopextractor import get_recommended_template_sizes

import numpy as np

def test_get_recommended_template_sizes():
    recommended_size = get_recommended_template_sizes(np.zeros((10,100,1000)))
    assert np.all(recommended_size == [3,15,63])

# FIXME: Add more tests