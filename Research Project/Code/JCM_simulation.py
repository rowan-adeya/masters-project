from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
from scipy.constants import k as k_B

# constants
w = 1.0
g = 1.0
N = 2

# Bases
basis_atom_e = q.basis(2, 0)
basis_qho_0 = q.basis(N, 0)


# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()


psi0 = q.tensor(basis_atom_e, basis_qho_0)

# Jaynes-Cummings Hamiltonian
H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)

# Closed Dynamics
sim_closed = TLSQHOSimulator(H, psi0)
results_closed = sim_closed.evolve()
vne_closed = sim_closed.vne(results_closed)
coherence_closed = sim_closed.rel_coherence(results_closed)

# Open Dynamics (Spontaneous Emission)
gamma = np.sqrt(0.1)

L_tls = gamma * s_lower
times_open = np.linspace(0.0, 500.0, 200)
e_ops = [adag * a, s_raise * s_lower]
sim_open_tls = TLSQHOSimulator(H, psi0, L_tls, e_ops, times_open)
results_open_tls = sim_open_tls.evolve()
expect_open_tls = sim_open_tls.expect(results_open_tls)

# Open Dynamics (QHO Photon Loss)
gamma_th = 0.1
T = 300  # avg T in Kelvin
n_omega = 1 / (np.exp(w / (k_B * T)) - 1)
kappa = gamma_th * (1 + n_omega)

L_qho = kappa * a
sim_open_qho = TLSQHOSimulator(H, psi0, L_qho, e_ops, times_open)
results_open_qho = sim_open_qho.evolve()
expect_open_qho = sim_open_qho.expect(results_open_qho)

L_tlsqho = [L_qho, L_tls]
sim_open_tlsqho = TLSQHOSimulator(H, psi0, L_tlsqho, e_ops, times=np.linspace(0.0, 2000.0, 200))
results_open_tlsqho = sim_open_tlsqho.evolve()
expect_open_tlsqho = sim_open_tlsqho.expect(results_open_tlsqho)


# Plots
sim_closed.plot(
    vne_closed,
    "Closed System Evolution of the JCM: Entanglement Oscillations",
    "Entanglement",
    "JCM_CQS_ent",
)
sim_closed.plot(
    coherence_closed,
    "Closed System Evolution of the JCM: Coherence Oscillations",
    "Coherence",
    "JCM_CQS_coh",
)

sim_open_tls.plot(
    expect_open_tls,
    "Open System Evolution of the JCM: Dynamics of Spontaneous Atomic Emission",
    "Expectation Values",
    "JCM_OQS_TLS_Decay",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_qho.plot(
    expect_open_qho,
    "Open System Evolution of the JCM: Dynamics of QHO Photon Loss",
    "Expectation Values",
    "JCM_OQS_QHO_loss",
    ["cavity photon number", "atom excitation probability"],
)

sim_open_tlsqho.plot(
    expect_open_tlsqho,
    "Open System Evolution of the JCM: Dynamics of TLS, QHO Decay",
    "Expectation Values",
    "JCM_OQS_QHOTLS",
    ["cavity photon number", "atom excitation probability"],
)
