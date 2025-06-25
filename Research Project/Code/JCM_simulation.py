from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np

# constants
w = 1.0
g = 1.0
N = 2
gamma = np.sqrt(0.1)

# Bases
basis_atom_g = q.basis(2, 1)
basis_atom_e = q.basis(2, 0)
basis_qho_0 = q.basis(N, 0)
basis_qho_1 = q.basis(N, 1)

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

# Open Dynamics
L = gamma * s_lower
times_open = np.linspace(0.0, 500.0, 200)
e_ops = [adag * a, s_raise * s_lower]
sim_open = TLSQHOSimulator(H, psi0, L, e_ops, times_open)
results_open = sim_open.evolve()
expect_open = sim_open.expect(results_open)


# Plots
sim_closed.plot(
    vne_closed,
    "Entanglement Oscillations For Closed Dynamics in the JCM",
    "Entanglement",
    "JCM_CQS_ent",
)
sim_closed.plot(
    coherence_closed,
    "Coherence Oscillations For Closed Dynamics in the JCM",
    "Coherence",
    "JCM_CQS_coh",
)

sim_open.plot(
    expect_open,
    "Expectation Values of Atom and QHO for Open Dynamics in the JCM",
    "Expectation Values",
    "JCM_OQS_expect",
    ["cavity photon number", "atom excitation probability"]
)
