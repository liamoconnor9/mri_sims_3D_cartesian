
import os
import numpy as np
import matplotlib.pyplot as plt

from dedalus import public as de
from dedalus.extras import plot_tools


def vp_bvp_func(domain, by, bz, bx):
    problem = de.LBVP(domain, variables=['Ax','Ay', 'Az', 'phi'])

    problem.parameters['by'] = by
    problem.parameters['bz'] = bz
    problem.parameters['bx'] = bx

    problem.add_equation("phi = 0", condition="(ny==0) and (nz==0)")
    problem.add_equation("Ax = 0", condition="(ny==0) and (nz==0)")
    problem.add_equation("Ay = 0", condition="(ny==0) and (nz==0)")
    problem.add_equation("Az = 0", condition="(ny==0) and (nz==0)")

    problem.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_equation("dy(Az) - dz(Ay) + dx(phi) = bx", condition="(ny!=0) or (nz!=0)")
    problem.add_equation("dz(Ax) - dx(Az) + dy(phi) = by", condition="(ny!=0) or (nz!=0)")
    problem.add_equation("dx(Ay) - dy(Ax) + dz(phi) = bz", condition="(ny!=0) or (nz!=0)")

    problem.add_bc("left(Ay) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("left(Az) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("right(Ay) = 0", condition="(ny!=0) or (nz!=0)")
    problem.add_bc("right(Az) = 0", condition="(ny!=0) or (nz!=0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    # Plot solution
    Ay = solver.state['Ay']
    Az = solver.state['Az']
    Ax = solver.state['Ax']

    return Ay, Az, Ax