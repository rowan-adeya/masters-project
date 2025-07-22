from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math


############################## SETUP ######################################

# NCOMMS4012: Consider pigments-protein environment of PE545 complex. Taking parameters stated,
# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = w*(0.0578)^0.5 = 267 cm^-1
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

########################## OPEN SIM SETUP #################################

# Constants
gamma = 0.1 
gamma_th = 0.1 
T = 300
kbT = 1.0 / (0.695 * T)
n_omega = 1 / (np.exp(w / kbT) - 1)

# Operators
e_ops = [adag * a, s_raise * s_lower]

L_tls = [np.sqrt(gamma) * s_lower]
L_qho = [
    np.sqrt(gamma_th) * (n_omega + 1) * a,  # photon loss
    np.sqrt(gamma_th) * n_omega * adag,  # photon gain
]

############################ SIMULATION ###################################

# TLS
sim_open_tls = TLSQHOSimulator(H, psi0, L_tls, e_ops, times=tlist)
results_open_tls = sim_open_tls.evolve()
expect_open_tls = sim_open_tls.expect(results_open_tls)
neg_tls = sim_open_tls.negativity(results_open_tls)
coh_tls = sim_open_tls.rel_coherence(results_open_tls)

# QHO
sim_open_qho = TLSQHOSimulator(H, psi0, L_qho, e_ops, times=tlist)
results_open_qho = sim_open_qho.evolve()
expect_open_qho = sim_open_qho.expect(results_open_qho)
neg_qho = sim_open_qho.negativity(results_open_qho)
coh_qho = sim_open_qho.rel_coherence(results_open_qho)

# TLS + QHO
L_tlsqho = L_tls + L_qho
sim_open_tlsqho = TLSQHOSimulator(H, psi0, L_tlsqho, e_ops, times=tlist)
results_open_tlsqho = sim_open_tlsqho.evolve()
expect_open_tlsqho = sim_open_tlsqho.expect(results_open_tlsqho)
neg_tlsqho = sim_open_tlsqho.negativity(results_open_tlsqho)
coh_tlsqho = sim_open_tlsqho.rel_coherence(results_open_tlsqho)

############################### PLOTS #####################################
sim_open_tls.plot(
    expect_open_tls,
    "Open Evolution ExVib: Spontaneous Atomic Emission (Weak Coupling)",
    "Expectation Values",
    "OQS_TLS_Decay",
    "ExVib",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tls.plot(
    neg_tls,
    "Open Evolution ExVib: Negativity of Atomic Emission (Weak Coupling)",
    "Negativity",
    "OQS_TLS_Neg",
    "ExVib",
    ["Negativity"],
)

sim_open_tls.plot(
    coh_tls,
    "Open Evolution ExVib: Coherence of Atomic Emission (Weak Coupling)",
    "Coherence",
    "OQS_TLS_coh",
    "ExVib",
    ["Coherence"],
)

sim_open_qho.plot(
    expect_open_qho,
    "Open Evolution ExVib: QHO Photon Loss/Gain (Weak Coupling)",
    "Expectation Values",
    "OQS_QHO_loss",
    "ExVib",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tls.plot(
    neg_qho,
    "Open Evolution ExVib: Negativity of QHO Photon Loss/Gain (Weak Coupling)",
    "Negativity",
    "OQS_QHO_Neg",
    "ExVib",
    ["Negativity"],
)

sim_open_qho.plot(
    coh_qho,
    "Open Evolution ExVib: Coherence of QHO Photon Loss/Gain (Weak Coupling)",
    "Coherence",
    "OQS_QHO_coh",
    "ExVib",
    ["Coherence"],
)

sim_open_tlsqho.plot(
    expect_open_tlsqho,
    "Open Evolution ExVib: TLS and QHO Decay (Weak Coupling)",
    "Expectation Values",
    "OQS_TLSQHO",
    "ExVib",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tls.plot(
    neg_tlsqho,
    "Open Evolution ExVib: Negativity of both Dissipators Acting",
    "Negativity",
    "OQS_TLSQHO_Neg",
    "ExVib",
    ["Negativity"],
)

sim_open_tls.plot(
    coh_tlsqho,
    "Open Evolution ExVib: Coherence of both Dissipators Acting",
    "Coherence",
    "OQS_TLSQHO_coh",
    "ExVib",
    ["Coherence"],
)