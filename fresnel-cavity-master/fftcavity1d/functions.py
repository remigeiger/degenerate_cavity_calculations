import math
import numpy as np


def round_to_odd(n):
    n = int(math.ceil(n))
    return n + 1 if n % 2 == 0 else n


def focused_linspace(start, stop, focuses, widths,
                     num_focused=100, num_unfocused=20):
    '''
    This function creates a linspace with `num_unfocused` points and then
    adds `num_focused` points around each `focus` with the given `width`
    '''
    try:
        if len(widths) == 1:
            widths = widths * len(focuses)
    except TypeError:
        widths = [widths] * len(focuses)
    out = np.linspace(start, stop, num_unfocused)
    for focus, width in zip(focuses, widths):
        if focus < start or stop < focus:
            raise ValueError('focuses should be between start and stop')
        idx_begin, idx_end = np.searchsorted(out,
                                             [focus - width / 2,
                                              focus + width / 2])
        out = np.hstack([out[:idx_begin],
                         np.linspace(focus - width / 2,
                                     focus + width / 2,
                                     num_focused),
                         out[idx_end:]])
    return out
