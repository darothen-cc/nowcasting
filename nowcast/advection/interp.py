""" Interpolation routines for use with the semi-Lagrangian advection
solver.

"""

from numba import njit
from numpy import isnan, floor, ceil

import numpy as np

#: A nominal background value which should be viable for most interpolation
#: tasks
BACKGROUND = 0.

@njit(nogil=True)
def access_array(arr, i, j, bkg=BACKGROUND):
    """ Safely access an arbitrary index from a 2D array.

    Parameters
    ----------
    arr : numpy.ndarray of rank 2
        The data array to access
    i, j : int
        Integer indices in the x- (columns) and y-directions (rows), respectively
    bkg : float
        A backup, "safety" value to return.

    Returns
    -------
    Array accessed at index (i, j), or the background value if this is out of bounds

    """

    nx, ny = arr.shape

    if (
        (i < 0) or (i >= nx) or
        (j < 0) or (j >= ny) or
        (isnan(arr[i, j]))
    ):
        return bkg
    else:
        return arr[i, j]


@njit(nogil=True)
def replace_nan(val, bkg=BACKGROUND):
    """ Return value or a safety (background) value if value is NaN. """

    if isnan(val):
        return bkg

    else:
        return val


@njit(nogil=True)
def bilinear_interp(arr, src_x, src_y, bkg=BACKGROUND):
    """ Perform bilinear interpolation to a fractional pixel location
    into an array.

    Parameters
    ----------
    arr : numpy.ndarray of rank 2
        The data array to access
    src_{x, y} : float
        Fractional indices in the x- (columns) and y-directions (rows) to
        interpolate to
    bkg : float
        A backup, "safety" value to return

    Returns
    -------
    Field represented by array, accessed at coordinates corresponding to the
    fractional multi-index (src_x, src_y)

    """

    nx, ny = arr.shape

    # Integer coordinates of the new source bin
    src_i = int(floor(src_x))
    src_j = int(floor(src_y))

    # Determine fractional contributions of each pixel under the source mask
    frac_i = 1.0 + src_i - src_x
    frac_j = 1.0 + src_j - src_y

    # If any of the source pixel mask is outside domain do the slow case
    if ((src_i < 0) or (src_i >= (nx - 1)) or
            (src_j < 0) or (src_j >= (ny - 1))):
        # If all the source pixels are outside, just set to border value
        if ((src_i < -1) or (src_i >= nx) or
                (src_j < -1) or (src_j >= ny)):
            return bkg
        else:
            # At least one good pixel.
            ret_val = (
                frac_i * frac_j * access_array(arr, src_i, src_j, bkg)
              + frac_i * (1.0 - frac_j) * access_array(arr, src_i, src_j + 1, bkg)
              + (1.0 - frac_i) * frac_j * access_array(arr, src_i + 1, src_j, bkg)
              + (1.0 - frac_i) * (1.0 - frac_j) * access_array(arr, src_i + 1, src_j + 1, bkg)
            )
    else:
        # Normal case - use all 4 pixels.
        ret_val = (
            frac_i * frac_j * replace_nan(arr[src_i, src_j], bkg)
          + frac_i * (1.0 - frac_j) * replace_nan(arr[src_i, src_j + 1], bkg)
          + (1.0 - frac_i) * frac_j * replace_nan(arr[src_i + 1, src_j], bkg)
          + (1.0 - frac_i) * (1.0 - frac_j) * replace_nan(arr[src_i + 1, src_j + 1], bkg)
        )

    return ret_val


#: Weights for bicubic interpolation
WT = np.array([
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0,
    2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1,
    0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1,
    -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
    9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2,
    -6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2,
    2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
    -6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1,
    4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1
]).astype(float).reshape((16, 16))


@njit("f8[:,:](f8[:], f8[:], f8[:], f8[:], f8, f8)", nogil=True)
def _bcucof(y, y1, y2, y12, d1, d2):
    """ Given arrays y, y1, y2, y12, each of length 4, containing
    the function, gradients, and cross derivative at the four grid points
    of a rectangular grid cell (numbered counterclockwise from lower left),
    and given d1 and d2, the length of the grid cell in the 1- and 2- directions,
    return the 4x4 table c necessary for bcuint to perform bicubic interpolation.

    This has been adapted from the similarly named subroutine in the F90
    release of Numerical Recipes, v2

    """

    # Pack a temporary vector x
    x = np.ones(16)
    x[:4] = y
    x[4:8] = y1 * d1
    x[8:12] = y2 * d2
    x[12:16] = y12 * d1 * d2

    # Matrix multiple the stored table
    c = WT @ x

    # Unpack the result into the output table
    c = c.reshape((4, 4))

    return c


@njit("f8(f8[:], f8[:], f8[:], f8[:], f8, f8, f8, f8, f8, f8)", nogil=True)
def _bcuint(y, y1, y2, y12, x1l, x1u, x2l, x2u, x1, x2):
    """ Bicubic interpolation within a grid square.

    Input quantities are y,y1,y2,y12 (as described in bcucof);
    x1l and x1u, the lower and upper coordinates of the grid square
    in the 1- direction; x2l and x2u likewise for the 2-direction;
    and x1,x2, the coordinates of the desired point for the interpolation.
    The interpolated function value is returned as ansy, and the
    interpolated gradient values as ansy1 and ansy2.

    This routine calls bcucof.

    This has been adapted from the similarly named subroutine in the F90
    release of Numerical Recipes, v2

    """

    # Get the c's
    c = _bcucof(y, y1, y2, y12, x1u - x1l, x2u - x2l)
    # print('c :', c)

    # Translate coordinates follow NRv2 Equation (3.6.4)
    t = (x1 - x1l) / (x1u - x1l)
    # print('t :', t, x1, x1l, x1u)
    u = (x2 - x2l) / (x2u - x2l)
    # print('u :', u)

    ansy = 0.
    # The variables ansy{1,2} are the x/y gradients at the interpolated point.
    # We don't really care about them, but I've kept them here for legacy purposes.
    # ansy2 = 0.
    # ansy1 = 0.
    for i in range(3, -1, -1):
        ansy = t * ansy + ((c[i, 3] * u + c[i, 2]) * u + c[i, 1]) * u + c[i, 0]
        # ansy2 = t*ansy2 + (3.*c[i,3]*u + 2.*c[i,2])*u + c[i,1]
        # ansy1 = u*ansy1 + (3.*c[3,i]*t + 2.*c[2,i])*t + c[1,i]
    # ansy1 = ansy1 / (x1u - x1l)
    # ansy2 = ansy2 / (x2u - x2l)

    return ansy  # , ansy1, ansy2


# @njit("f8(f8[:], f8[:], f8[:,:], f8, f8, f8, f8)", nogil=True)
def bicubic_interp(x, y, field, xp, yp):
    """ Perform bicubic interpolation on a regularly-gridded, 2D dataset.

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively
    field : 2D array
        The tracer field to perform the interpolation over
    xp, yp : floats
        Point at which to compute the interpolated value of `field`

    Returns
    -------
    field interpolated to (xp, yp)

    """

    # Compute grid-spacing in both directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute the fractional integer index of the point in question
    i1, j1 = (xp - x[0]) / dx, (yp - y[0]) / dy

    # Set up the interpolation square
    ilo, ihi = int(floor(i1)), int(ceil(i1))
    jlo, jhi = int(floor(j1)), int(ceil(j1))
    xlo, xhi = x[ilo], x[ihi]
    ylo, yhi = y[jlo], y[jhi]

    # Corner case - this should only happen if we're in upper-left
    # corner of the grid, but it will eventually lead to a divide by
    # zero error. So we should just return the corner grid cell value.
    if ((ilo == ihi) or (jlo == jhi)):
        return field[0, 0]

    dfield_dx = (field[2:, :] - field[:-2, :]) / 2. / dx
    dfield_dy = (field[:, 2:] - field[:, :-2]) / 2. / dy
    dfield_dxdy = (dfield_dx[:, 2:] - dfield_dx[:, :-2]) / 2. / dy

    # Loop through the corners, counter-clockwise starting from the bottom left
    _y, _y1, _y2, _y12 = np.ones(4), np.ones(4), np.ones(4), np.ones(4)
    ct = 0
    corners = [(ilo, jlo), (ihi, jlo), (ihi, jhi), (ilo, jhi)]
    for _i, _j in corners:
        _y[ct] = field[_i, _j]
        # Note the dimensionality reduction in the arrays below -
        # we used centered differences, which cleaved off a halo of
        # width 1, so we need to add that back in when we index into
        # the derivative arrays
        _y1[ct] = dfield_dx[_i - 1, _j]
        _y2[ct] = dfield_dy[_i, _j - 1]
        _y12[ct] = dfield_dxdy[_i - 1, _j - 1]
        ct += 1

    return _bcuint(_y, _y1, _y2, _y12, xlo, xhi, ylo, yhi, xp, yp)


# @njit("f8(f8[:], f8[:], f8[:,:],  f8[:,:], f8[:,:], f8[:,:], f8, f8, f8, f8)", nogil=True)
def bicubic_interp2(x, y, field, dfx, dfy, dfxy, xp, yp):
    """ Similar to bicubic_interp but requires the derivatives to be passed to
    the routine (not computed on the fly)

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively
    field : 2D array
        The tracer field to perform the interpolation over
    dfx, dfy, dfxy : 2D arrays
        Gradients of the tracer field in the x, y, and x-y cross dimensions,
        respectively. We assume that these gradients were computed using
        centered finite differences, such that if `field` is a n x m array,
        then these arrays have dimensions (n-2) x (m-2)
    xp, yp : floats
        Point at which to compute the interpolated value of `field`

    Returns
    -------
    field interpolated to (xp, yp)

    """

    # Compute grid-spacing in both directions
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute the fractional integer index of the point in question
    i1, j1 = (xp - x[0]) / dx, (yp - y[0]) / dy

    # Set up the interpolation square
    ilo, ihi = int(floor(i1)), int(ceil(i1))
    jlo, jhi = int(floor(j1)), int(ceil(j1))
    xlo, xhi = x[ilo], x[ihi]
    ylo, yhi = y[jlo], y[jhi]

    # Corner case - this should only happen if we're in upper-left
    # corner of the grid, but it will eventually lead to a divide by
    # zero error. So we should just return the corner grid cell value.
    if ( (ilo == ihi) or (jlo == jhi) ):
        return field[0, 0]

    # Loop through the corners, counter-clockwise starting from the bottom left
    _y, _y1, _y2, _y12 = np.ones(4), np.ones(4), np.ones(4), np.ones(4)
    ct = 0
    corners = [(ilo, jlo), (ihi, jlo), (ihi, jhi), (ilo, jhi)]
    for _i, _j in corners:
        _y[ct] = field[_i, _j]
        # Note the dimensionality reduction in the arrays below -
        # we used centered differences, which cleaved off a halo of
        # width 1, so we need to add that back in when we index into
        # the derivative arrays
        _y1[ct] = dfx[_i - 1, _j]
        _y2[ct] = dfy[_i, _j - 1]
        _y12[ct] = dfxy[_i - 1, _j - 1]
        ct += 1

    return _bcuint(_y, _y1, _y2, _y12, xlo, xhi, ylo, yhi, xp, yp)