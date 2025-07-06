from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np


############################## SETUP ######################################

tlist = np.linspace(0.0, 100.0, 200)

# constants
w = 1.0  # natural units, w = 1, hbar =1, kb = 1, GHz freq
g = 0.1
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


############################ SIMULATION ###################################

sim_closed = TLSQHOSimulator(H, psi0, times=tlist)
results_closed = sim_closed.evolve()
vne_closed = sim_closed.vne(results_closed)
coherence_closed = sim_closed.rel_coherence(results_closed)


############################### PLOTS #####################################

sim_closed.plot(
    vne_closed,
    "Closed System Evolution of the JCM: Entanglement Oscillations",
    "Entanglement",
    "CQS_ent",
    "JCM",
)
sim_closed.plot(
    coherence_closed,
    "Closed System Evolution of the JCM: Coherence Oscillations",
    "Coherence",
    "CQS_coh",
    "JCM",
)
