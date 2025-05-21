import numpy as np

#===========================
grid_shape = (1024,1024,1024)
#grid_shape = (512,512,512)
#grid_shape = (256,256,256)
#grid_shape = (128,128,128)
#grid_shape = (64,64,64)

#proc_shape = (1,1,1)
#proc_shape = (8,8,1)
proc_shape = (16,16,1)
halo_shape = 1

#===========================
lambda0 = 0.2
v = 1/61
g = np.sqrt(lambda0/2)

Tc = np.sqrt(3)*v
T0 = 2.4*Tc

gstar = 106.75
rho_r = gstar*np.pi**2/30 * T0**4
a0 = 1
Hubble_0 = np.sqrt(8*np.pi*rho_r/3)

fstar = v
wstar = a0 * Hubble_0


def potential(f, T):
    phi_re, phi_im = f[0], f[1]
    V = 1/4 * lambda0 * fstar**4 * ((phi_re**2+phi_im**2)-v**2/fstar**2)**2 + \
        (lambda0/6 + g**2/4) * T**2 * (phi_re**2+phi_im**2) * fstar**2
    return V

m_eff = np.sqrt((lambda0 * (1/3 + g**2/(2*lambda0)) * T0**2 - v**2))

#===========================
delta_x = 0.05
delta_t = 0.01

L = delta_x * grid_shape[0]
box_dim = (L, L, L)
volume = np.product(box_dim)
dx = tuple(Li / Ni for Li, Ni in zip(box_dim, grid_shape))
dk = tuple(2 * np.pi / (Li) for Li in box_dim)
dt = delta_t

#===========================
dtype = np.float64
nscalars = 2
mpl = 1

step_interval = 50
#steptosave_string = [i for i in range(1,10000,step_interval)]
steptosave_string = [10000000000]

gw_step_interval = 10
energy_step_interval = 10
gw_begin = 18

f0 = [0., 0.] 
df0 = [0., 0.]

start_time = 1
end_step = 6000

