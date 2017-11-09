
from clawpack import riemann
from clawpack import pyclaw

import numpy as np
import xarray as xr

def advect_field(u, v, qfunc, qkws={},
                 nx=200, ny=50, dx=500., dy=500.,
                 t_out=np.arange(0, 3601, 5*60),
                 sharp=True):
    """ Run a 2D advection calculation via clawpack.

    """

    is_vc = isinstance(u, (np.ndarray, ))

    if is_vc:
        rp = riemann.vc_advection_2D
    else:
        rp = riemann.advection_2D

    if sharp:
        solver = pyclaw.SharpClawSolver2D(rp)
    else:
        solver = pyclaw.ClawSolver2D(rp)
        solver.limiters = pyclaw.limiters.tvd.vanleer

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic
    solver.bc_lower[1] = pyclaw.BC.periodic
    solver.bc_upper[1] = pyclaw.BC.periodic

    if is_vc:
        solver.aux_bc_lower[0] = pyclaw.BC.periodic
        solver.aux_bc_upper[0] = pyclaw.BC.periodic
        solver.aux_bc_lower[1] = pyclaw.BC.periodic
        solver.aux_bc_upper[1] = pyclaw.BC.periodic

    # Register domain
    x = pyclaw.Dimension(0, dx*nx, nx, name='x')
    y = pyclaw.Dimension(0, dy*ny, ny, name='y')
    domain = pyclaw.Domain([x, y])

    x1d = domain.grid.x.centers
    y1d = domain.grid.y.centers
    xx = domain.grid.c_centers[0]
    yy = domain.grid.c_centers[1]

    num_eqn = 1
    state = pyclaw.State(domain, num_eqn)

    if is_vc:
        state.aux[0, ...] = u
        state.aux[1, ...] = v
    else:
        state.problem_data['u'] = u # m/s
        state.problem_data['v'] = v # m/s

    q = qfunc(xx, yy, **qkws)
    state.q[0, ...] = q

    claw = pyclaw.Controller()
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.keep_copy = True

    claw.tfinal = t_out[-1]
    claw.out_times = t_out
    claw.num_output_times = len(t_out) - 1

    claw.run()

    times = claw.out_times
    tracers = [f.q.squeeze() for f in claw.frames]
    tracers = np.asarray(tracers)

    if not is_vc:
        u = u*np.ones_like(xx)
        v = v*np.ones_like(xx)

    print(tracers.shape, u.shape, xx.shape, x1d.shape)

    ds = xr.Dataset(
        {'q': (('time', 'x', 'y'), tracers),
         'u': (('x', 'y'), u),
         'v': (('x', 'y'), v)},
        {'time': times, 'x': x1d, 'y': y1d},
    )
    ds['time'].attrs.update({'long_name': 'time', 'units': 'seconds since 2000-01-01 0:0:0'})
    ds['x'].attrs.update({'long_name': 'x-coordinate', 'units': 'm'})
    ds['y'].attrs.update({'long_name': 'y-coordinate', 'units': 'm'})
    ds['u'].attrs.update({'long_name': 'zonal wind', 'units': 'm/s'})
    ds['v'].attrs.update({'long_name': 'meridional wind', 'units': 'm/s'})
    ds.attrs.update({
        'Conventions': 'CF-1.7'
    })

    return ds


def advect_1d(u, qfunc, qkws={},
              nx=200, dx=500.,
              t_out=np.arange(0, 3601, 5*60),
              sharp=True):
    """ Run a 1D advection calculation via clawpack. CONSTANT U ONLY.

    """
    rp = riemann.advection_1D

    if sharp:
        solver = pyclaw.SharpClawSolver1D(rp)
        solver.weno_order = 5
    else:
        solver = pyclaw.ClawSolver1D(rp)

    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic

    x = pyclaw.Dimension(0, dx*nx, nx, name='x')
    domain = pyclaw.Domain(x)

    state = pyclaw.State(domain, solver.num_eqn)
    state.problem_data['u'] = u

    x1d = domain.grid.x.centers
    q = qfunc(x1d, t=0)

    state.q[0, ...] = q

    claw = pyclaw.Controller()
    claw.keep_copy = True
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver

    claw.tfinal = t_out[-1]
    claw.out_times = t_out
    claw.num_output_times = len(t_out) - 1

    claw.run()

    times = claw.out_times
    tracers = [f.q.squeeze() for f in claw.frames]
    tracers = np.asarray(tracers)

    uarr = u*np.ones_like(x1d)

    ds = xr.Dataset(
        {'q': (('time', 'x'), tracers),
         'u': (('x', ), uarr)},
        {'time': times, 'x': x1d}
    )
    ds['time'].attrs.update({'long_name': 'time', 'units': 'seconds since 2000-01-01 0:0:0'})
    ds['x'].attrs.update({'long_name': 'x-coordinate', 'units': 'm'})
    ds['u'].attrs.update({'long_name': 'zonal wind', 'units': 'm/s'})
    ds.attrs.update({
        'Conventions': 'CF-1.7'
    })

    return ds
