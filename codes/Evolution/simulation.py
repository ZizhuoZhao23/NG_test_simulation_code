import numpy as np
import pyopencl as cl
import pyopencl.array as cla
import pystella as ps
import time
from mpi4py import MPI

import setup
import field_init
import leap_frog_evolve as LP_evolve

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()
#==============================================
grid_shape = setup.grid_shape
proc_shape = setup.proc_shape
rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape,proc_shape))
grid_size = grid_shape[0] * grid_shape[1] * grid_shape[2]
halo_shape = setup.halo_shape
pencil_shape = tuple(ni + 2 * halo_shape for ni in rank_shape)

#=============================================
potential = setup.potential
m_eff = setup.m_eff
T0 = setup.T0

gstar = setup.gstar
g = setup.g
lambda0 = setup.lambda0
v = setup.v

a0 = setup.a0
Hubble0 = setup.Hubble_0

fstar = setup.fstar
wstar = setup.wstar

dx = setup.dx
dk = setup.dk
dt = setup.dt
L = setup.L
volume = setup.volume

gw_step_interval = setup.gw_step_interval
energy_step_interval = setup.energy_step_interval
gw_begin = setup.gw_begin

steptosave_string = setup.steptosave_string
f0 = setup.f0
df0 = setup.df0

start_time = setup.start_time
end_step = setup.end_step

#==============================================
dtype = setup.dtype
nscalars = setup.nscalars

Stepper = ps.LowStorageRK54
gravitational_waves = True

ctx = ps.choose_device_and_make_context()
queue = cl.CommandQueue(ctx)

decomp = ps.DomainDecomposition(proc_shape, halo_shape, rank_shape)
fft = ps.DFT(decomp, ctx, queue, grid_shape, dtype)
if halo_shape == 0:
    derivs = ps.SpectralCollocator(fft, dk)
else:
    derivs = ps.FiniteDifferencer(decomp, halo_shape, dx, rank_shape=rank_shape)

#==============================================
if gravitational_waves:
    gw_sector = ps.TensorPerturbationSector(nscalars)
    sectors = [gw_sector]

    stepper = Stepper(sectors, halo_shape=halo_shape, rank_shape=rank_shape, dt=dt)

#==============================================

if decomp.rank == 0:
    from pystella.output import OutputFile
    out = OutputFile(ctx=ctx, runfile=__file__)
else:
    out = None

statistics = ps.FieldStatistics(decomp, halo_shape, rank_shape=rank_shape,
                                grid_size=grid_size)
spectra = ps.PowerSpectra(decomp, fft, dk, volume)
projector = ps.Projector(fft, halo_shape, dk, dx)
hist = ps.FieldHistogrammer(decomp, 1000, dtype, rank_shape=rank_shape)

#==============================================
def output(step_count, t, a, Hubble, f, dhijdt, energy_scalar, energy_gauge):
    
    if step_count in steptosave_string:
        all_f = cla.empty(queue, (nscalars,)+grid_shape, dtype)
        decomp.gather_array(queue, f[0], all_f[0], 0)
        decomp.gather_array(queue, f[1], all_f[1], 0)
        
        if decomp.rank == 0:
            out.output('field', field=all_f.get())

    spec_out = {}
    if gravitational_waves and step_count % gw_step_interval == 0:
        Hnow = np.array(Hubble).astype(dtype) / wstar
        a = a
        spec_out['gw_scalar'] = spectra.gw(dhijdt[0], projector, Hnow)
        spec_out['gw_gauge'] = spectra.gw(dhijdt[1], projector, Hnow)
        spec_out['gw_all'] = spectra.gw(dhijdt[0]+dhijdt[1], projector, Hnow)
        
        test_1 = decomp.allreduce(dhijdt.get())
        if decomp.rank == 0:
            print(np.sum(test_1))
            
        if decomp.rank == 0:
            out.output('spectra', t=t, a=a, **spec_out)

#==============================================
# Initialization
w = 1/3
s1 = slice(halo_shape,halo_shape+rank_shape[0])
s2 = slice(halo_shape,halo_shape+rank_shape[1])
s3 = slice(halo_shape,halo_shape+rank_shape[2])
normal_s = (s1, s2, s3)

LP_evo = LP_evolve.Evolution(queue, decomp, pencil_shape, rank_shape, dtype, dx, dt, start_time, 
                             lambda0, v, g, a0, Hubble0, T0, wstar, fstar, w, gstar, normal_s)
 
if gravitational_waves:
    hij = cla.empty(queue, (2,)+(6,)+pencil_shape, dtype) * 0
    dhijdt = cla.empty(queue, (2,)+(6,)+pencil_shape, dtype) * 0
    lap_hij = cla.empty(queue, (2,)+(6,)+rank_shape, dtype) * 0
else:
    hij, dhijdt, lap_hij = None, None, None

f = np.zeros((nscalars,)+pencil_shape, dtype=np.float64)
dfdt = np.zeros((nscalars,)+pencil_shape, dtype=np.float64)

im_E = np.zeros((3,)+pencil_shape, dtype=np.float64)
re_E = np.zeros((3,)+pencil_shape, dtype=np.float64) + 2/dx[0]/dt/g

A = np.zeros((3,)+pencil_shape, dtype=np.float64)
V = np.ones((3,)+pencil_shape, dtype=np.complex128)
#-----------------------
a = a0
Hubble = Hubble0
T = T0


tmp_f = cla.empty(queue, (2,)+pencil_shape, dtype)
tmp_dfdt = cla.empty(queue, (2,)+pencil_shape, dtype)

modes = ps.RayleighGenerator(ctx, fft, dk, volume, seed=49279*(decomp.rank+1))
for i in range(2):
    modes.init_WKB_fields(tmp_f[i], tmp_dfdt[i], queue=queue, norm=wstar**2/fstar**2, omega_k=lambda k: np.sqrt(k**2 + m_eff**2/wstar**2), 
                          field_ps=lambda wk: 1/wk * 1/(np.exp(wk/T0*wstar)-1), hubble=Hubble/wstar)
f = tmp_f.get()
dfdt = tmp_dfdt.get()

for i in range(2):
    decomp.share_halos(queue, f[i])
    decomp.share_halos(queue, dfdt[i])


phi = f[0] + 1j * f[1]
pi = dfdt[0] + 1j * dfdt[1]
im_E, re_E = field_init.init_gauge_field(phi, pi, grid_shape, pencil_shape, fft, g, dx, dt, norm=fstar**2/wstar**2)

for d in range(3):
    decomp.share_halos(queue, im_E[d])
    decomp.share_halos(queue, re_E[d])
    
#==============================================
output(0, 0, a0, Hubble0, f, dhijdt, 0, 0)

#==============================================
t = start_time
t_phy = start_time
step_count = 0

phi_amp_bar = 1.

#===================================
LP_evo.initialization(f, dfdt, im_E, V)

#==============================================
while step_count < end_step and (1-np.isnan(phi_amp_bar)):
    time_begin = time.time()
    #-----------------------------------
    t += dt
    step_count += 1
    
    LP_evo.a_evolve(t)
    t_phy += LP_evo.a * dt

    LP_evo.momentum_evolve()
    LP_evo.Gauss_constraint()
    LP_evo.field_evolve()

    #----------------------------------
    LP_evo.gw_terms()

    #----------------------------------
    if gravitational_waves and t > gw_begin:
        for s in range(stepper.num_stages):
            stepper(s, queue=queue, hubble=np.array(LP_evo.Hubble),
                    scalar=LP_evo.scalar_gw, gauge=LP_evo.gauge_gw, 
                    hijs=hij[0], dhijsdt=dhijdt[0], lap_hijs=lap_hij[0],
                    hijg=hij[1], dhijgdt=dhijdt[1], lap_hijg=lap_hij[1], filter_args=True)
            derivs(queue, fx=hij, lap=lap_hij)
    time5 = time.time()

    #----------------------------------
    if step_count % energy_step_interval == 1:
        LP_evo.energy_calculate(grid_size, potential, step_count)
        LP_evo.energy_polarization(queue, fft, projector, dk, grid_shape[0])
        
    #-----------------------------------
    output(step_count, t, LP_evo.a, LP_evo.Hubble, LP_evo.f, dhijdt, LP_evo.energy_scalar, LP_evo.energy_gauge)

    #=================================
    if step_count % energy_step_interval == 1:
        phi_re_sum, phi_im_sum = decomp.allreduce(LP_evo.f[0][normal_s]), decomp.allreduce(LP_evo.f[1][normal_s])
        phi_re_bar, phi_im_bar = np.sum(phi_re_sum)/grid_size, np.sum(phi_im_sum)/grid_size
        phi_bar = (phi_re_bar**2 + phi_im_bar**2)**(1/2)

        phi_amp = (LP_evo.f[0]**2 + LP_evo.f[1]**2)**(1/2)
        phi_amp_sum = decomp.allreduce(phi_amp[normal_s])
        phi_amp_bar = np.sum(phi_amp_sum)/grid_size
        
        Ax, Ay, Az = decomp.allreduce(LP_evo.A[0][normal_s]), decomp.allreduce(LP_evo.A[1][normal_s]), decomp.allreduce(LP_evo.A[2][normal_s])
        Ax_bar, Ay_bar, Az_bar = np.sum(Ax)/grid_size, np.sum(Ay)/grid_size, np.sum(Az)/grid_size
        

        GW_source_scalar, GW_source_gauge = LP_evo.scalar_gw.get(), LP_evo.gauge_gw.get()
        GW_source_scalar = decomp.allreduce(GW_source_scalar)
        GW_source_gauge = decomp.allreduce(GW_source_gauge)
        GW_source_scalar_bar = np.sum(GW_source_scalar)/grid_size
        GW_source_gauge_bar = np.sum(GW_source_gauge)/grid_size

        Gauss_minus = np.sum(decomp.allreduce(np.abs(LP_evo.Gauss_minus))) / grid_size
        Gauss_plus = np.sum(decomp.allreduce(np.abs(LP_evo.Gauss_plus))) / grid_size
        Gauss_constraint = Gauss_minus / Gauss_plus

        
        if decomp.rank == 0:
            out.output('a', t=t, a=LP_evo.a, Hubble=LP_evo.Hubble)
            out.output('T', t=t, T=LP_evo.T)
            out.output('phi', t=t, phi_re=phi_re_bar, phi_im=phi_im_bar, phi_amp=phi_amp_bar)
            out.output('A', t=t, Ax=Ax_bar, Ay=Ay_bar, Az=Az_bar)
            out.output('energy', t=t, kinetic=LP_evo.kinetic_scalar_bar, gradient=LP_evo.gradient_scalar_bar,
                                potential=LP_evo.energy_potential_bar,
                                electric=LP_evo.electric_gauge_bar, magnetic=LP_evo.magnetic_gauge_bar)
            out.output('energy_rad', t=t, kinetic=LP_evo.rad_kinetic_scalar_bar, gradient=LP_evo.rad_gradient_scalar_bar,
                                electric=LP_evo.rad_electric_gauge_bar, magnetic=LP_evo.rad_magnetic_gauge_bar)
            out.output('particle_energy', t=t, 
                                scalar=LP_evo.tot_scalar_energy,  gauge_L=LP_evo.tot_L_energy, gauge_T=LP_evo.tot_T_energy, 
                                n_s=LP_evo.n_s, n_A_L=LP_evo.n_A_L, n_A_T=LP_evo.n_A_T,
                                n_s_spectrum=LP_evo.n_s_spectrum, n_A_L_spectrum=LP_evo.n_A_L_spectrum, n_A_T_spectrum=LP_evo.n_A_T_spectrum, 
                                rho_s_spectrum=LP_evo.rho_s_spectrum, rho_A_L_spectrum=LP_evo.rho_A_L_spectrum, rho_A_T_spectrum=LP_evo.rho_A_T_spectrum)
            
            out.output('Gauss_constraint', t=t, Gauss=Gauss_minus, Gauss_rescale=Gauss_constraint)
            out.output('GW_source', t=t, scalar=GW_source_scalar_bar, gauge=GW_source_gauge_bar)
            
    time_end = time.time()
    if decomp.rank == 0:
        print('step:',step_count, 't:', t, 't_phy:', t_phy)
        print('a:', LP_evo.a, 'Hubble:', LP_evo.Hubble, 'T:', LP_evo.T)
        print()
        print('Ax:', Ax_bar, 'Ay:', Ay_bar, 'Az:', Az_bar)
        print('phi_re:', phi_re_bar, 'phi_im:', phi_im_bar)
        print('phi_amp:', phi_amp_bar, 'phi_bar:', phi_bar)
        print()
        print('Gauss constraint:', Gauss_constraint, 'plus:', Gauss_plus, 'minus:', Gauss_minus)
        print()
        print('energy kinetic:', LP_evo.kinetic_scalar_bar, 'energy gradient:', LP_evo.gradient_scalar_bar, 'energy potential:', LP_evo.energy_potential_bar)
        print('energy electric:', LP_evo.electric_gauge_bar, 'energy magnetic:', LP_evo.magnetic_gauge_bar)
        print()
        print('scalar particle:', LP_evo.tot_scalar_energy, 'gauge L particle:', LP_evo.tot_L_energy, 'gauge T particle:', LP_evo.tot_T_energy)
        print('ns:', LP_evo.n_s*LP_evo.a**3/v**3, 'nA_L:', LP_evo.n_A_L*LP_evo.a**3/v**3, 'nA_T:', LP_evo.n_A_T*LP_evo.a**3/v**3)
        print()
        print('time:', time_end-time_begin)
        print('===============================================')
        print()
    
