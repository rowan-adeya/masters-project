from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math

# https://opg.optica.org/josab/fulltext.cfm?uri=josab-41-8-C206&id=553027
# https://ar5iv.labs.arxiv.org/html/2112.08917v1
# links for weak/strong regimes
############################## SETUP ######################################

tlist = np.linspace(0.0, 100.0, 200)

# constants
w = 1.0  # natural units, w = 1, hbar =1, kb = 1, GHz freq
g = 0.1
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


psi0 = q.tensor(basis_atom_e + basis_atom_g, basis_qho) / math.sqrt(2)

# Jaynes-Cummings Hamiltonian
H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)


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
    "JCM",
    ["cavity photon number", "atom excitation probability"],
)

sim_closed.plot(
    vne_closed,
    "Closed Evolution JCM: Entanglement",
    "Entanglement",
    "CQS_ent_eg",
    "JCM",
)
sim_closed.plot(
    coherence_closed,
    "Closed Evolution JCM: Coherence",
    "Coherence",
    "CQS_coh_eg",
    "JCM",
)
