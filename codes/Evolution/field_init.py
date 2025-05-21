import numpy as np
import pyopencl.array as cla

def init_gauge_field(phi, pi, grid_shape, pencil_shape, fft, g, dx, dt, norm):

    kshape = fft.shape(True)
    J_k = np.zeros(kshape, dtype=np.complex128)
    J = -np.imag(pi.conj()*phi)
    J = J.copy()
    fft.dft(J, J_k)

    sub_k = list(x.get() for x in fft.sub_k.values())
    kvecs = np.meshgrid(*sub_k, indexing="ij", sparse=False)
    
    N = grid_shape[0]
    tmp = np.sin(np.pi*kvecs[0]/N)**2 + np.sin(np.pi*kvecs[1]/N)**2 + np.sin(np.pi*kvecs[2]/N)**2
    tmp[0,0,0] = 1
    Y_i = 1j * g * dx[0] * 1/tmp * np.sin(np.pi*kvecs[0]/N) * J_k
    Y_j = 1j * g * dx[1] * 1/tmp * np.sin(np.pi*kvecs[1]/N) * J_k
    Y_k = 1j * g * dx[2] * 1/tmp * np.sin(np.pi*kvecs[2]/N) * J_k
    
    Y_i *= np.exp(1j * np.pi*kvecs[0]/N) / N**3 * norm
    Y_j *= np.exp(1j * np.pi*kvecs[1]/N) / N**3 * norm
    Y_k *= np.exp(1j * np.pi*kvecs[2]/N) / N**3 * norm
    my_Y = [Y_i, Y_j, Y_k]

    im_E = np.zeros((3,)+pencil_shape)
    for i in range(3):
        fft.idft(my_Y[i], im_E[i])
    re_E = ((1/(dx[0]*dt*g)**2 - im_E**2))**(1/2)

    return im_E, re_E
