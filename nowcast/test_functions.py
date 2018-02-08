""" Test functions for running idealized advection simulations. """

import numpy as np

COORD_DTYPE = np.dtype([('x', float), ('y', float)])

################################################################################
# Velocity fields

def eddy(x, y, A=4.):
    """ Create a 2D flow-field comprised of a single, large eddy centered in
    the domain.

    .. math::
        \begin{align*}
            u(x, y) &= Ay \\
            v(x, y) &= -Ax
        \end{align*}

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively (m)
    A : float
        Maximum velocity of flow field (m/s)

    Returns
    -------
    u and v arrays containing the zonal and meridional components of the flow

    """

    xf, yf = x[-1, 0], y[0, -1]

    u = A*(2.*(y - yf/2.)/yf)
    v = -A*(2.*(x - x[-1]/2.)/xf)

    return u, v


def wavy_flow(x, y, Ax=1., Ay=1., Tx=2.):
    """

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively (m)
    A{x, y} : float
        Maximum velocity of flow field in x- and y- directions (m/s)
    Tx : float
        Number of vertical waves in the zonal direction

    Returns
    -------
    u and v arrays containing the zonal and meridional components of the flow

    """

    x_dist = x[-1, 0] - x[0, 0]

    u = Ax*np.ones_like(y)
    v = Ay*np.sin(Tx*2.*np.pi*x/x_dist)

    return u, v


################################################################################
# Initial conditions

def gaussian(x, y, A=1., x0=0., y0=0., sigmax=1., sigmay=1., offset=0.):
    """ Produce a 2D Gaussian hill with given amplitude 'A', centers at
    'x0' and 'y0', background constant offset 'offset', and decay
    rates 'sigmax' and 'sigmay'

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively
    A : float
        Maximum value of hill, excluding the offset
    x0, y0 : float
        coordinates of center of gaussian hill
    sigmax, sigmay : float
        width of distribution in the x- and y-directions
    offset : float
        background constant to add to all data points

    Returns
    -------
    2D array with the same shape as x and y

    """
    exp_l = (x - x0)**2 / (2.*sigmax*sigmax)
    exp_r = (y - y0)**2 / (2.*sigmay*sigmay)
    return offset + A * np.exp(-1.*(exp_l + exp_r))


def min_gaussian(x, y, minimum=0.01, **kws):
    """ Same as `gaussian`, but threshold all the values the given field
    before returning.

    """
    q = gaussian(x, y, **kws)
    q[q < minimum] = 0.0
    return q


def gauss_1d(x, t, x0, L, u=1, A=1):
    """ 1D Gaussian bell of amplitude A, standard deviation L, and translating
    center starting at x0 but moving with a velocity of u, sampled at time t.

    """
    return A*np.exp(-((x - x0 - (u*t))**2)/L/L)


def slotted_cylinder(x, y, cx, cy, r, A=10.):
    """ Create a slotted cylinder.

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively
    A : float
        Maximum value of hill, excluding the offset
    cx, cy : float
        Coordinates for center of the cylinder
    r : float
        Radius of the cylinder
    A : float
        Amplitude of the cylinder

    Returns
    -------
    2D array with the slotted cylinder initial conditions

    """

    F = np.zeros_like(x)
    dx = r / 4.

    # Cylinder
    F[((x-cx)**2 + (y-cy)**2) < r**2] = A

    # Slot
    F[(x-(cx-dx) > 0) & (x-(cx+dx) < 0) & (y-cy > 0)] = 0.

    return F


def hat(x, y, A, xbnds=[0, 1], ybnds=[0, 1]):
    """ Compute a 2D "hat" or plateau".

    Parameters
    ----------
    x, y : 2D array
        2D arrays containing the x- and y-coordinates of the data grid points,
        respectively
    A : float
        Amplitude/height of hat
    xbnds, ybnds : 2-tuple
        Coordinate bounds which define the hat

    Returns
    -------
    2D array with the same shape as x and y

    """
    xlo, xhi = xbnds
    ylo, yhi = ybnds
    mask = (xlo < x) & (x < xhi) & (ylo < y) & (y < yhi)
    data = np.zeros_like(x)
    data[mask] = A
    return data


def hat_1d(x, t, x0, dx, u=1, A=1):
    """ 1D hat of amplitude A and width dx, and translating center starting
    at x0 ut moving with a velocity of u, sampled at time t.

    """
    y = np.zeros_like(x)
    mask = (x > (x0 - dx + u*t)) & (x < (x0 + dx + u*t))
    y[mask] = A
    return y


def multi_wave(x, y, nx=2, ny=1):
    """ Compute an arbitrary zonally/meridionally varying wave.

    .. math::
        Z = \cos{\frac{n_x \lambda}{T_\lambda}} +
            2\frac{\phi - \bar{\phi}}{\mathrm{std}(\phi)}
    Parameters
    ----------
    lons, lats : array-like of floats
        Longitude/latitude coordinate at which to evaluate wave equation
    nx, ny : int
        Wavenumber in zonal and meridional direction
    """
    Tx = x.max() / nx
    Ty = y.max() / ny
    fx, fy = 1./Tx, 1./Ty
    return np.cos(2.*np.pi*fx*x) + 2*(y - np.mean(y))/y.std()
