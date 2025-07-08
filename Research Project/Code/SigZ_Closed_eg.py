from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math

############################## SETUP ######################################

tlist = np.linspace(0.0, 100.0, 200)

# constants
w = 1.0
g = 0.1  # Weak coupling
N = 30  # num Fock states
n = 2  # photon number

# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho = q.basis(N, n)


# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()


#  Hamiltonian with SigmaZ Interaction
H = (
    0.5 * w * s_z
    + w * (adag * a + 0.5)
    + g * q.tensor(-1 * q.sigmaz(), q.destroy(N).dag() + q.destroy(N))
)

# Init cond
psi0 = q.tensor(basis_atom_e + basis_atom_g, basis_qho) / math.sqrt(2)
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
    "Closed Evolution JCM: Expectation Values",
    "Expectation Values",
    "CQS_expt_eg",
    "SigZ",
    ["cavity photon number", "atom excitation probability"],
)

sim_closed.plot(
    vne_closed,
    "Closed Evolution SigmaZ: Interaction Entanglement",
    "Entanglement",
    "CQS_ent_eg",
    "SigZ",
    legend=None,
)
sim_closed.plot(
    coherence_closed,
    "Closed Evolution SigmaZ: Coherence",
    "Coherence",
    "CQS_coh_eg",
    "SigZ",
    legend=None,
)
