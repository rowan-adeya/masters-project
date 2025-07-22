from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math


############################## SETUP ######################################

# NCOMMS4012: Consider pigments-protein environment of PE545 complex. Taking parameters stated,
# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = (0.0578)^0.5 = 267 cm^-1
# Moreover, T ~ 300K room temp expt.

t_max = 10.0 # ps
tlist = np.linspace(0.0, 2 *np.pi * 0.03 * t_max , 500)  
# TODO : conversion within class tlist_cm = tlist * 2 *np.pi * 3**(-2)  
# t in cm conversion, noting original time is in ps

# params, cm^-1
w = 1111
d_epsilon = 1042
V = 92
g = w * math.sqrt(0.0578)
# weak regime, g < gamma, gamma_th AND g << w (zeta = 0.01, C = 1)
N = 10  # num Fock states
n = 0  # photon number


# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho = q.basis(N, n)


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
e_ops = [adag * a, s_raise * s_lower]

sim_closed = TLSQHOSimulator(H, psi0, e_ops=e_ops, times=tlist)
results_closed = sim_closed.evolve()
results_expt = sim_closed.expect(results_closed)
vne_closed = sim_closed.vne(results_closed)
coherence_closed = sim_closed.rel_coherence(results_closed)


############################### PLOTS #####################################
sim_closed.plot(
    results_expt,
    "Closed Evolution ExVib: Expectation Values",
    "Expectation Values",
    "CQS_expt",
    "ExVib",
    ["cavity photon number", "atom excitation probability"],
)

sim_closed.plot(
    vne_closed,
    "Closed Evolution ExVib: Interaction Entanglement",
    "Entanglement",
    "CQS_ent",
    "ExVib",
    legend=None,
)
sim_closed.plot(
    coherence_closed,
    "Closed Evolution ExVib: Coherence",
    "Coherence",
    "CQS_coh",
    "ExVib",
    legend=None,
)
