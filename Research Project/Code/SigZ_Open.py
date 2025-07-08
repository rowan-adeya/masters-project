from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np


############################## SETUP ######################################

tlist = np.linspace(0.0, 50.0, 2000)

# constants
w = 1.0
g = 0.05  # weak regime, g < gamma, gamma_th AND g << w (zeta = 0.01, C = 1)
N = 30 # num Fock states
n = 2 # photon number 

# Bases
basis_atom_e = q.basis(2, 0)
basis_qho_0 = q.basis(N, n)


# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()


psi0 = q.tensor(basis_atom_e, basis_qho_0)

# Jaynes-Cummings Hamiltonian
H = (
    0.5 * w * s_z
    + w * (adag * a + 0.5)
    + g * q.tensor(-1 * q.sigmaz(), q.destroy(N).dag() + q.destroy(N))
)

########################## OPEN SIM SETUP #################################

# Constants
gamma = 0.1
gamma_th = 0.1
kbT = 0.001
n_omega = 1 / (np.exp(w / kbT) - 1)

# Operators
e_ops = [adag * a, s_raise * s_lower]

L_tls = [gamma * s_lower]
L_qho = [
    gamma_th * (n_omega + 1) * a,  # photon loss
    gamma_th * n_omega * adag,  # photon gain
]

############################ SIMULATION ###################################

# TLS
sim_open_tls = TLSQHOSimulator(H, psi0, L_tls, e_ops, times=tlist)
results_open_tls = sim_open_tls.evolve()
expect_open_tls = sim_open_tls.expect(results_open_tls)
neg_tls = sim_open_tls.negativity(results_open_tls)

# QHO
sim_open_qho = TLSQHOSimulator(H, psi0, L_qho, e_ops, times=tlist)
results_open_qho = sim_open_qho.evolve()
expect_open_qho = sim_open_qho.expect(results_open_qho)
neg_qho = sim_open_qho.negativity(results_open_qho)

# TLS + QHO
L_tlsqho = L_tls + L_qho
sim_open_tlsqho = TLSQHOSimulator(H, psi0, L_tlsqho, e_ops, times=tlist)
results_open_tlsqho = sim_open_tlsqho.evolve()
expect_open_tlsqho = sim_open_tlsqho.expect(results_open_tlsqho)

############################### PLOTS #####################################
sim_open_tls.plot(
    expect_open_tls,
    "Open Evolution SigmaZ: Spontaneous Atomic Emission (Weak Coupling)",
    "Expectation Values",
    "OQS_TLS_Decay",
    "SigZ",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tls.plot(
    neg_tls,
    "Open Evolution SigmaZ: Negativity of Atomic Emission (Weak Coupling)",
    "Negativity",
    "OQS_TLS_Neg",
    "SigZ",
    ["Negativity"],
)

sim_open_qho.plot(
    expect_open_qho,
    "Open Evolution SigmaZ: QHO Photon Loss/Gain (Weak Coupling)",
    "Expectation Values",
    "OQS_QHO_loss",
    "SigZ",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tls.plot(
    neg_qho,
    "Open Evolution SigmaZ: Negativity of QHO Photon Loss (Weak Coupling)",
    "Negativity",
    "OQS_QHO_Neg",
    "SigZ",
    ["Negativity"],
)
sim_open_tlsqho.plot(
    expect_open_tlsqho,
    "Open Evolution SigmaZ: TLS and QHO Decay (Weak Coupling)",
    "Expectation Values",
    "OQS_QHOTLS",
    "SigZ",
    ["cavity photon number", "atom excitation probability"],
)
