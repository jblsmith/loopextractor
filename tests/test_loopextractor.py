from loopextractor.loopextractor import get_recommended_template_sizes, purify_core_tensor, choose_bar_to_reconstruct

import numpy as np

def test_get_recommended_template_sizes():
    recommended_size = get_recommended_template_sizes(np.zeros((10,100,1000)))
    assert np.all(recommended_size == [3,15,63])


def test_purify_cube():
    thing1 = np.random.randint(0, 1000, (5,5))
    thing2 = np.random.randint(0, 1000, (5,5))
    thing3 = np.random.randint(0, 1000, (5,5))
    thing4 = thing1 + 2*thing2
    factors = [[], [], np.array([[1,0], [1,0], [0,1], [0,1]]).T]
    core_tensor = np.dstack([thing1, thing2, thing3, thing4])
    pure_tensor, pure_factors = purify_core_tensor(core_tensor, factors, new_rank=3, dim_to_reduce=2)
    assert pure_tensor.shape == (5,5,3)

    bar_0_before = np.dot(core_tensor, factors[2][0, :])
    bar_0_after = np.dot(pure_tensor, pure_factors[2][0, :])
    assert np.all(np.abs(bar_0_before - bar_0_after) < bar_0_before)


def test_choose_bar_to_reconstruct():
    # Three bars, 4 loops
    loop_map = np.array([[11,10,1,0,12],
                         [0,3,2,1,1],
                         [5,0,0,1,2]]).T
    assert choose_bar_to_reconstruct(loop_map, 0) == 4
    assert choose_bar_to_reconstruct(loop_map, 1) == 1
    assert choose_bar_to_reconstruct(loop_map, 2) == 0

# FIXME: Add more tests