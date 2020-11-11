#!/usr/bin/env python3

import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve

def kdeHist(x, y, xbins=30, ybins=23,
             xlim=[-3.0,0], ylim=[-1.25,1]):
    mask_x = np.isfinite(x)
    mask_y = np.isfinite(y)
    x_m = x[mask_x & mask_y]
    y_m = y[mask_x & mask_y]
    counts, _xbins, _ybins = np.histogram2d(x_m, y_m, bins=[xbins, ybins],
                                          density=True,
                                          range=[xlim,ylim])
    xx, yy = np.mgrid[xlim[0]:xlim[1]:complex(0,xbins),
                      ylim[0]:ylim[1]:complex(0,ybins)]
    # xx, yy = np.meshgrid(xbins, ybins, indexing='ij')
    return xx, yy, counts
