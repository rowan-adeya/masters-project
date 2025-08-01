from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math


############################## SETUP ######################################

# NCOMMS4012: Consider pigments-protein environment of PE545 complex. Taking parameters stated,
# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = (0.0578)^0.5 = 267 cm^-1
# Moreover, T ~ 300K room temp expt.

t_max = 45.0  # ps
tlist = np.linspace(0.0, 2 * np.pi * 0.03 * t_max, 100)
# TODO : conversion within class tlist_cm = tlist * 2 *np.pi * 3**(-2)
# t in cm conversion, noting original time is in ps

# params, cm^-1
w = 1111
d_epsilon = 1042
V = 92
g = w * math.sqrt(0.0578)
# weak regime, g < gamma, gamma_th AND g << w (zeta = 0.01, C = 1)
N = 10  # num Fock states

# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho = q.basis(N, 0)


# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_x = q.tensor(q.sigmax(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()

# Exciton-Vibration Hamiltonian
H_ex = (0.5 * d_epsilon * s_z) + (V * s_x)
H_vib = w * adag * a
H_int = -(g / math.sqrt(2)) * q.tensor(q.sigmaz(), q.destroy(N).dag() + q.destroy(N))
H = H_ex + H_vib + H_int

# Initial Condition
psi0 = q.tensor(basis_atom_e, basis_qho)

############################ SIMULATION ###################################
n0_pop = q.tensor(q.qeye(2), basis_qho * basis_qho.dag())  # QHO pop |n=0>
e_ops = [s_lower * s_raise, n0_pop]

sim = TLSQHOSimulator(H, psi0, e_ops=e_ops, times=tlist)
results = sim.evolve()

results_expt = sim.expect(results)

vne = sim.vne(results)

coh_tls = sim.rel_coherence(results, subsys="TLS")
coh_qho = sim.rel_coherence(results, subsys="QHO")


############################### PLOTS #####################################

sim.plot(
    "pops_excited",
    [
        {
            "y_data": results_expt[0],
            "label": "Exciton ground state",
            "colour": "tab:blue",
        },
        {
            "y_data": results_expt[1],
            "label": "Vibration photon number",
            "colour": "tab:red",
        },
    ],
    y_label="Expectation Values",
    savepath="/ExVib/Closed",
)

sim.plot(
    "vne",
    [{"y_data": vne}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="/ExVib/Closed",
)

sim.plot(
    "coh",
    [
        {"y_data": coh_tls, "label": "Exciton subsystem", "colour": "tab:blue"},
        {"y_data": coh_qho, "label": "Vibration subsystem", "colour": "tab:red"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="/ExVib/Closed",
)
