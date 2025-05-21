import numpy as np
import pyopencl.array as cla
import numexpr as ne

def tensor_index(i, j):
    a = i if i <= j else j
    b = j if i <= j else i
    return (7 - a) * a // 2 - 4 + b

#================================
class Evolution():
    def __init__(self, queue, decomp, pencil_shape, rank_shape, dtype, dx, dt, start_time, 
                lambda0, v, g, a0, Hubble0, T0, wstar, fstar, w, gstar, normal_s):
        self.queue = queue
        self.decomp = decomp
        self.pencil_shape = pencil_shape
        self.rank_shape = rank_shape
        self.dtype = dtype
        self.start_time = start_time

        self.dx, self.dt = dx, dt
        self.lambda0, self.v, self.g = lambda0, v, g
        self.a0, self.Hubble0, self.T0 = a0, Hubble0, T0
        self.wstar, self.fstar = wstar, fstar
        self.p = 2 / (3*(1+w)-2)
        self.gstar = gstar
        
        self.normal_s = normal_s

    #============================
    def adjust_slices(self, axis, offset=0, s=None):
        if s == None:
            slices = list(self.normal_s)
        else:
            slices = list(s)

        target_slice = slices[axis]
        adjusted_slice = slice(target_slice.start + offset,
                               target_slice.stop + offset)
        slices[axis] = adjusted_slice
        return tuple(slices)
    
    #============================
    def initialization(self, f, dfdt, im_E, V):
        self.f, self.dfdt = f, dfdt
        self.V, self.im_E = V, im_E
        self.a_half_m = self.a0 * (1 + 1/self.p * self.a0*self.Hubble0 * (-self.dt/2)/self.wstar)**self.p

        #-----------------------------------
        self.phi = f[0] + 1j * f[1]
        self.pi = dfdt[0] + 1j * dfdt[1]
        
        term_1 = 0
        for i in range(3):
            s_shift_ip = self.adjust_slices(i, 1)
            s_shift_im = self.adjust_slices(i, -1)
            term_1 += ne.evaluate("V * phi", {'V': self.V[i][self.normal_s], 'phi': self.phi[s_shift_ip]})
            term_1 += ne.evaluate("V * phi", {'V': self.V[i].conj()[s_shift_im], 'phi': self.phi[s_shift_im]})
            term_1 -= 2 * self.phi[self.normal_s]
            
        term_1 *= self.a0**2 / self.dx[0]**2
        term_2 = -self.a0**4 * (1/2 * self.lambda0 * (self.fstar/self.wstar)**2 * \
                                (ne.evaluate("phi * phi_conj", {'phi': self.phi[self.normal_s], 'phi_conj': self.phi.conj()[self.normal_s]})
                                 - self.v**2/self.fstar**2) * self.phi[self.normal_s] + \
                                (self.lambda0/6 + self.g**2/4) * 1/self.wstar**2 * self.T0**2 * self.phi[self.normal_s])

        scalar1, scalar2 = np.zeros((2,)+self.rank_shape), np.zeros((2,)+self.rank_shape)
        scalar1[0], scalar2[0] = np.real(term_1), np.real(term_2)
        scalar1[1], scalar2[1] = np.imag(term_1), np.imag(term_2)
        
        #-----------------------------------
        term_1 = np.zeros((3,)+self.rank_shape, dtype=np.complex128)
        term_2 = np.zeros((3,)+self.rank_shape, dtype=np.complex128)
        for i in range(3):
            s_shift_ip = self.adjust_slices(i, 1)
            term_1[i] = ne.evaluate("phi_conj * V * phi", {'phi_conj': self.phi.conj()[s_shift_ip], 'V': self.V[i].conj()[self.normal_s], 'phi': self.phi[self.normal_s]})
            for j in range(3):
                s_shift_ip = self.adjust_slices(i, 1)
                s_shift_jp = self.adjust_slices(j, 1)
                s_shift_jm = self.adjust_slices(j, -1)
                s_shift_jm_ip = self.adjust_slices(i, 1, self.adjust_slices(j, -1))
                term_2[i] += ne.evaluate("V1*V2*V3*V4", {'V1': self.V[i][self.normal_s], 'V2': self.V[j][s_shift_ip], 'V3': self.V[i].conj()[s_shift_jp], 'V4': self.V[j].conj()[self.normal_s]})
                term_2[i] += ne.evaluate("V1*V2*V3*V4", {'V1': self.V[j][s_shift_jm], 'V2': self.V[i][self.normal_s], 'V3': self.V[j].conj()[s_shift_jm_ip], 'V4': self.V[i].conj()[s_shift_jm]})
                
        term_1 *= 2*self.g / self.dx[0] * (self.fstar/self.wstar)**2 * self.a0**2
        term_2 *= -1 / self.g / self.dx[0]**3

        link1, link2 = np.zeros((3,)+self.rank_shape), np.zeros((3,)+self.rank_shape)
        for d in range(3):
            link1[d], link2[d] = np.imag(term_1[d]), np.imag(term_2[d])
        
        #-----------------------------------
        self.dfdt_half = np.zeros((2,)+self.rank_shape)
        self.im_E_half = np.zeros((3,)+self.rank_shape)
        
        for i in range(2):
            self.dfdt_half[i] = self.dfdt[i][self.normal_s]*(self.a0/self.a_half_m)**2 - \
                            self.dt/2 / self.a_half_m**2 * (scalar1[i] + scalar2[i])
        for d in range(3):
            self.im_E_half[d] = self.im_E[d][self.normal_s] - self.dt/2 * (link1[d] + link2[d])
        self.re_E_half = ((1/(self.dx[0]*self.dt*self.g)**2 - self.im_E_half**2))**(1/2)

        self.dfdt_half_old = self.dfdt_half.copy()
        self.im_E_half_old = self.im_E_half.copy()

    #============================
    def a_evolve(self, t):
        self.a_half_m = self.a0 * (1 + 1/self.p * self.a0*self.Hubble0 * (t-self.dt/2 - self.start_time) / self.wstar)**self.p
        self.a_half_p = self.a0 * (1 + 1/self.p * self.a0*self.Hubble0 * (t+self.dt/2 - self.start_time) / self.wstar)**self.p
        self.a = 1/2 * (self.a_half_p + self.a_half_m)
        self.Hubble = self.a0**2*self.Hubble0 * (1 + 1/self.p * self.a0*self.Hubble0 * (t - self.start_time) / self.wstar)**(self.p-1) / self.a
        self.T = (self.Hubble**2/self.a**2 * 3/(8*np.pi) * 30/(self.gstar*np.pi**2))**(1/4)
    
    #============================
    def momentum_evolve(self):
        term_1 = 0
        for i in range(3):
            s_shift_ip = self.adjust_slices(i, 1)
            s_shift_im = self.adjust_slices(i, -1)
            term_1 += ne.evaluate("V * phi", {'V': self.V[i][self.normal_s], 'phi': self.phi[s_shift_ip]})
            term_1 += ne.evaluate("V * phi", {'V': self.V[i].conj()[s_shift_im], 'phi': self.phi[s_shift_im]})
            term_1 -= 2 * self.phi[self.normal_s]

        term_1 *= self.a**2 / self.dx[0]**2
        term_2 = -self.a**4 * (1/2 * self.lambda0 * (self.fstar/self.wstar)**2 * \
                                (ne.evaluate("phi * phi_conj", {'phi': self.phi[self.normal_s], 'phi_conj': self.phi.conj()[self.normal_s]})
                                 - self.v**2/self.fstar**2) * self.phi[self.normal_s] + \
                                (self.lambda0/6 + self.g**2/4) * 1/self.wstar**2 * self.T**2 * self.phi[self.normal_s])

        scalar1, scalar2 = np.zeros((2,)+self.rank_shape), np.zeros((2,)+self.rank_shape)
        scalar1[0], scalar2[0] = np.real(term_1), np.real(term_2)
        scalar1[1], scalar2[1] = np.imag(term_1), np.imag(term_2)
        
        #-----------------------------------
        term_1 = np.zeros((3,)+self.rank_shape, dtype=np.complex128)
        term_2 = np.zeros((3,)+self.rank_shape, dtype=np.complex128)
        for i in range(3):
            s_shift_ip = self.adjust_slices(i, 1)
            term_1[i] = ne.evaluate("phi_conj * V * phi", {'phi_conj': self.phi.conj()[s_shift_ip], 'V': self.V[i].conj()[self.normal_s], 'phi': self.phi[self.normal_s]})
            for j in range(3):
                s_shift_ip = self.adjust_slices(i, 1)
                s_shift_jp = self.adjust_slices(j, 1)
                s_shift_jm = self.adjust_slices(j, -1)
                s_shift_jm_ip = self.adjust_slices(i, 1, self.adjust_slices(j, -1))
                term_2[i] += ne.evaluate("V1*V2*V3*V4", {'V1': self.V[i][self.normal_s], 'V2': self.V[j][s_shift_ip], 'V3': self.V[i].conj()[s_shift_jp], 'V4': self.V[j].conj()[self.normal_s]})
                term_2[i] += ne.evaluate("V1*V2*V3*V4", {'V1': self.V[j][s_shift_jm], 'V2': self.V[i][self.normal_s], 'V3': self.V[j].conj()[s_shift_jm_ip], 'V4': self.V[i].conj()[s_shift_jm]})
                
        term_1 *= 2*self.g / self.dx[0] * (self.fstar/self.wstar)**2 * self.a**2
        term_2 *= -1 / self.g / self.dx[0]**3

        link1, link2 = np.zeros((3,)+self.rank_shape), np.zeros((3,)+self.rank_shape)
        for d in range(3):
            link1[d], link2[d] = np.imag(term_1[d]), np.imag(term_2[d])
        
        #-----------------------------------
        for i in range(2):
            self.dfdt_half[i] = self.dfdt_half[i]*(self.a_half_m/self.a_half_p)**2 + \
                                self.dt/self.a_half_p**2 * (scalar1[i] + scalar2[i])
        for d in range(3):
            self.im_E_half[d] += self.dt * (link1[d] + link2[d])
        self.re_E_half = ((1/(self.dx[0]*self.dt*self.g)**2 - self.im_E_half**2))**(1/2)
        
        for i in range(2):
            self.dfdt[i][self.normal_s] = (self.dfdt_half_old[i] + self.dfdt_half[i]) / 2
            self.decomp.share_halos(self.queue, self.dfdt[i])
        for d in range(3):
            self.im_E[d][self.normal_s] = (self.im_E_half_old[d] + self.im_E_half[d]) / 2
            self.decomp.share_halos(self.queue, self.im_E[d])
        self.re_E = ((1/(self.dx[0]*self.dt*self.g)**2 - self.im_E**2))**(1/2)

        self.dfdt_half_old = self.dfdt_half.copy()
        self.im_E_half_old = self.im_E_half.copy()
        self.pi = self.dfdt[0] + 1j * self.dfdt[1]

    #============================
    def Gauss_constraint(self):
        term_1 = 0
        for i in range(3):
            s_shift_im = self.adjust_slices(i, -1)
            term_1 += self.im_E[i][self.normal_s] - self.im_E[i][s_shift_im]
            
        term_1 /= self.dx[0]
        term_2 = 2*self.g * self.a**2 * (self.fstar/self.wstar)**2 * \
                 np.imag(ne.evaluate("pi * phi", {'pi': self.pi.conj()[self.normal_s], 'phi': self.phi[self.normal_s]}))

        self.Gauss_minus = term_1 - term_2
        self.Gauss_plus = term_1 + term_2
        
    #============================
    def field_evolve(self):
        E_half = self.re_E_half + 1j * self.im_E_half
        for i in range(2):
            self.f[i][self.normal_s] += self.dt * self.dfdt_half[i]
            self.decomp.share_halos(self.queue, self.f[i])
        self.phi = self.f[0] + 1j * self.f[1]
        for d in range(3):
            self.V[d][self.normal_s] *= self.dx[0] * self.dt * self.g * E_half[d]
            self.decomp.share_halos(self.queue, self.V[d])
        self.A = -1 * np.angle(self.V) / self.g / self.dx[0]

    #============================
    def gw_terms(self):
        self.F = np.zeros((6,)+self.rank_shape)
        for i in range(1,4):
            for j in range(i,4):
                index = tensor_index(i, j)
                s_shift_jp = self.adjust_slices(j-1,1)
                s_shift_ip = self.adjust_slices(i-1,1)
                Vij = ne.evaluate("V1*V2*V3*V4", {'V1': self.V[j-1][self.normal_s], 'V2': self.V[i-1][s_shift_jp], 'V3': self.V[j-1].conj()[s_shift_ip], 'V4': self.V[i-1].conj()[self.normal_s]})
                self.F[index] = 1/(self.g*self.dx[0]**2) * np.imag(Vij)
        
        self.gauge_gw = np.zeros((6,)+self.rank_shape)
        for i in range(1,4):
            for j in range(i,4):
                index = tensor_index(i, j)
                self.gauge_gw[index] -= 1/self.a**2 * ne.evaluate("Ei * Ej", {'Ei': self.im_E[i-1][self.normal_s], 'Ej': self.im_E[j-1][self.normal_s]})
                for k in range(1,4):
                    index_11, index_12 = tensor_index(i, k), tensor_index(k, i)
                    index_21, index_22 = tensor_index(k, j), tensor_index(j, k)

                    F1 = self.F[index_11] if k>=i else -self.F[index_12]
                    F2 = self.F[index_21] if j>=k else -self.F[index_22]
                    self.gauge_gw[index] -= 1/self.a**2 * ne.evaluate("F1 * F2", {'F1': F1, 'F2': F2})

                    
        self.scalar_gw = np.zeros((6,)+self.rank_shape)
        for i in range(1,4):
            for j in range(i,4):
                index = tensor_index(i, j)
                s_shift_ip = self.adjust_slices(i-1,1)
                s_shift_jp = self.adjust_slices(j-1,1)
            
                dphi_i = (ne.evaluate("V * phi", {'V': self.V[i-1][self.normal_s], 'phi': self.phi[s_shift_ip]}) -\
                          self.phi[self.normal_s]) / self.dx[0]
                dphi_j = (ne.evaluate("V * phi", {'V': self.V[j-1][self.normal_s], 'phi': self.phi[s_shift_jp]}) -\
                          self.phi[self.normal_s]) / self.dx[0]
                self.scalar_gw[index] = 2 * np.real(ne.evaluate("phi * phi_conj", {'phi': dphi_i, 'phi_conj': dphi_j.conj()}))

        self.gauge_gw *= self.wstar**2
        self.scalar_gw *= self.fstar**2

        self.gauge_gw = cla.to_device(self.queue, self.gauge_gw)
        self.scalar_gw = cla.to_device(self.queue, self.scalar_gw)
    
    
    #============================
    def energy_calculate(self, grid_size, potential, step_count):
        self.rho = np.sqrt(np.real(ne.evaluate("phi * phi_conj", {'phi': self.phi, 'phi_conj': self.phi.conj()})))
        no_string_pos = self.rho**2
        no_string_pos = no_string_pos[self.normal_s]
        if step_count == 1:
            self.rho_old = self.rho
        
        #------------------------------------
        kinetic_scalar = np.real(ne.evaluate("pi * pi_conj", {'pi': self.pi[self.normal_s], 'pi_conj': self.pi.conj()[self.normal_s]}))
    
        gradient_scalar = 0
        for d in range(3):
            s_shift_dp = self.adjust_slices(d, 1)
            tmp_phi = (ne.evaluate("V * phi", {'V': self.V[d][self.normal_s], 'phi': self.phi[s_shift_dp]}) \
                       - self.phi[self.normal_s]) / self.dx[d]
            gradient_scalar += np.real(ne.evaluate("phi * phi_conj", {'phi': tmp_phi, 'phi_conj': tmp_phi.conj()}))

        kinetic_scalar *= self.fstar**2 * self.wstar**2 / self.a**2
        gradient_scalar *= self.fstar**2 * self.wstar**2 / self.a**2
        
        rad_kinetic_scalar = kinetic_scalar * no_string_pos
        rad_gradient_scalar = gradient_scalar * no_string_pos

        self.energy_scalar = kinetic_scalar + gradient_scalar
        self.energy_potential = potential(self.f, self.T)
        self.energy_potential = self.energy_potential[self.normal_s]

        kinetic_scalar_all = self.decomp.allreduce(kinetic_scalar)
        gradient_scalar_all = self.decomp.allreduce(gradient_scalar)
        energy_potential_all = self.decomp.allreduce(self.energy_potential)
        self.kinetic_scalar_bar = np.sum(kinetic_scalar_all)/grid_size
        self.gradient_scalar_bar = np.sum(gradient_scalar_all)/grid_size
        self.energy_potential_bar = np.sum(energy_potential_all)/grid_size

        kinetic_scalar_all = self.decomp.allreduce(rad_kinetic_scalar)
        gradient_scalar_all = self.decomp.allreduce(rad_gradient_scalar)
        self.rad_kinetic_scalar_bar = np.sum(kinetic_scalar_all)/grid_size
        self.rad_gradient_scalar_bar = np.sum(gradient_scalar_all)/grid_size
        
        #------------------------------------
        electric_gauge, magnetic_gauge = 0, 0
        for i in range(1,4):
            for j in range(1,4):
                index = tensor_index(i, j) if j>=i else tensor_index(j, i)
                magnetic_gauge += self.F[index]**2 / 4

        for i in range(3):
            electric_gauge += self.im_E[i][self.normal_s]**2 / 2
        
        electric_gauge *= self.wstar**4 / self.a**4
        magnetic_gauge *= self.wstar**4 / self.a**4

        rad_electric_gauge = electric_gauge * no_string_pos
        rad_magnetic_gauge = magnetic_gauge * no_string_pos

        self.energy_gauge = electric_gauge + magnetic_gauge

        electric_gauge_all = self.decomp.allreduce(electric_gauge)
        magnetic_gauge_all = self.decomp.allreduce(magnetic_gauge)
        self.electric_gauge_bar = np.sum(electric_gauge_all)/grid_size
        self.magnetic_gauge_bar = np.sum(magnetic_gauge_all)/grid_size

        electric_gauge_all = self.decomp.allreduce(rad_electric_gauge)
        magnetic_gauge_all = self.decomp.allreduce(rad_magnetic_gauge)
        self.rad_electric_gauge_bar = np.sum(electric_gauge_all)/grid_size
        self.rad_magnetic_gauge_bar = np.sum(magnetic_gauge_all)/grid_size


    #============================
    def energy_polarization(self, queue, fft, projector, dk, N):

        rho_s = np.real(self.phi.conj() * self.pi)
        rho_s = np.ascontiguousarray(rho_s)
        rho_s = cla.to_device(queue, rho_s)


        Ei = -cla.to_device(queue, self.im_E)
        Ei_phi = -cla.to_device(queue, self.im_E)
        rho = cla.to_device(queue, self.rho)
        for d in range(3):
            Ei_phi[d] *= rho

        kshape = fft.shape(True)
        cdtype = fft.cdtype

        rho_s_k = cla.empty(queue, kshape, cdtype, allocator=None)
        fft.dft(rho_s, rho_s_k)
        Ei_k = cla.empty(queue, (3,)+kshape, cdtype, allocator=None)
        for d in range(3):
            fft.dft(Ei[d], Ei_k[d])
        Ei_phi_k = cla.empty(queue, (3,)+kshape, cdtype, allocator=None)
        for d in range(3):
            fft.dft(Ei_phi[d], Ei_phi_k[d])

        plus = cla.empty(queue, kshape, cdtype, allocator=None)
        minus = cla.empty(queue, kshape, cdtype, allocator=None)
        lng = cla.empty(queue, kshape, cdtype, allocator=None)
        projector.decompose_vector(queue, Ei_phi_k, plus, minus, lng,
                                        times_abs_k=True)

        sub_k = list(x.get() for x in fft.sub_k.values())
        kvecs = np.meshgrid(*sub_k, indexing="ij", sparse=False)

        lng_G = 0
        tmp_Ei_k = Ei_k.get()
        for d in range(3):
            lng_G += tmp_Ei_k[d] * kvecs[d] * dk[d]
        
        tmp_rho_s_k = rho_s_k.get()
        scalar = tmp_rho_s_k * tmp_rho_s_k.conj() * self.fstar**2 * self.wstar**2 / self.a**2
        scalar = np.real(scalar)

        mA = np.sqrt(2*self.g**2) * self.v
        lng = lng.get()
        lng = lng_G*lng_G.conj() * 2*self.wstar**6/(mA**2*self.a**2) + \
            lng*lng.conj() * self.fstar**2 * self.wstar**4 / self.v**2
        lng = np.real(lng)

        plus, minus = plus.get(), minus.get()
        trans = (plus*plus.conj() + minus*minus.conj()) * self.fstar**2 * self.wstar**4 / self.v**2
        trans = np.real(trans)

        kmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(dk, kvecs)))
        if self.decomp.nranks > 1:
            from mpi4py import MPI
            max_k = self.decomp.allreduce(np.max(kmags), MPI.MAX)
        else:
            max_k = np.max(kmags)
        num_bins = int(max_k / min(dk) + .5) + 1
        bins = np.arange(-.5, num_bins + .5) * min(dk)
        k_bins = np.array([i*min(dk) for i in range(num_bins)])

        kmags_flat = kmags.flatten()
        scalar_flat = scalar.flatten()
        lng_flat = lng.flatten()
        trans_flat = trans.flatten()
        bin_indices = np.digitize(kmags_flat, bins) - 1
        bin_num = np.bincount(bin_indices, minlength=num_bins)

        I_scalar = np.bincount(bin_indices, weights=scalar_flat, minlength=num_bins)
        square_mean_scalar =  np.divide(I_scalar, bin_num, out=np.zeros_like(I_scalar), where=bin_num!=0)

        I_lng = np.bincount(bin_indices, weights=lng_flat, minlength=num_bins)
        square_mean_lng =  np.divide(I_lng, bin_num, out=np.zeros_like(I_lng), where=bin_num!=0)

        I_trans = np.bincount(bin_indices, weights=trans_flat, minlength=num_bins)
        square_mean_trans =  np.divide(I_trans, bin_num, out=np.zeros_like(I_trans), where=bin_num!=0)

        self.tot_scalar_energy = 0
        for i in range(len(I_scalar)):
            self.tot_scalar_energy += k_bins[i]**2 * square_mean_scalar[i] * dk[0]
        self.tot_scalar_energy *= 1/(2*np.pi**2) * (self.dx[0]/N)**3
        self.tot_scalar_energy = self.decomp.allreduce(self.tot_scalar_energy)

        ms = np.sqrt(self.lambda0) * self.v
        drho_sdk = k_bins**2 * square_mean_scalar * 1/(2*np.pi**2) * (self.dx[0]/N)**3
        E_s = np.sqrt((k_bins*self.wstar / self.a)**2 + ms**2)
        self.n_s = 0
        for i in range(len(drho_sdk)):
            self.n_s += dk[0] * drho_sdk[i] / E_s[i]
        self.n_s = self.decomp.allreduce(self.n_s)

        self.n_s_spectrum = drho_sdk / E_s * k_bins
        self.n_s_spectrum = self.decomp.allreduce(self.n_s_spectrum)

        self.rho_s_spectrum = drho_sdk
        self.rho_s_spectrum = self.decomp.allreduce(self.rho_s_spectrum)



        self.tot_L_energy = 0
        self.tot_T_energy = 0
        for i in range(len(I_lng)):
            self.tot_L_energy += k_bins[i]**2 * square_mean_lng[i] * dk[0]
            self.tot_T_energy += k_bins[i]**2 * square_mean_trans[i] * dk[0]
        self.tot_L_energy *= 1/(2*np.pi**2) * (self.dx[0]/N)**3 / self.a**4
        self.tot_T_energy *= 1/(2*np.pi**2) * (self.dx[0]/N)**3 / self.a**4
        self.tot_L_energy = self.decomp.allreduce(self.tot_L_energy)
        self.tot_T_energy = self.decomp.allreduce(self.tot_T_energy)

        drho_Ldk = k_bins**2 * square_mean_lng * 1/(2*np.pi**2) * (self.dx[0]/N)**3 / self.a**4
        drho_Tdk = k_bins**2 * square_mean_trans * 1/(2*np.pi**2) * (self.dx[0]/N)**3 / self.a**4
        E_A = np.sqrt((k_bins*self.wstar / self.a)**2 + mA**2)
        self.n_A_L, self.n_A_T = 0, 0
        for i in range(len(drho_Ldk)):
            self.n_A_L += dk[0] * drho_Ldk[i] / E_A[i]
            self.n_A_T += dk[0] * drho_Tdk[i] / E_A[i]
        self.n_A_L = self.decomp.allreduce(self.n_A_L)
        self.n_A_T = self.decomp.allreduce(self.n_A_T)

        self.n_A_L_spectrum = drho_Ldk / E_A * k_bins
        self.n_A_T_spectrum = drho_Tdk / E_A * k_bins
        self.n_A_L_spectrum = self.decomp.allreduce(self.n_A_L_spectrum)
        self.n_A_T_spectrum = self.decomp.allreduce(self.n_A_T_spectrum)

        self.rho_A_L_spectrum = drho_Ldk
        self.rho_A_T_spectrum = drho_Tdk
        self.rho_A_L_spectrum = self.decomp.allreduce(self.rho_A_L_spectrum)
        self.rho_A_T_spectrum = self.decomp.allreduce(self.rho_A_T_spectrum)
    

