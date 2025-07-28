from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np

# https://opg.optica.org/josab/fulltext.cfm?uri=josab-41-8-C206&id=553027
# https://ar5iv.labs.arxiv.org/html/2112.08917v1
# links for weak/strong regimes
############################## SETUP ######################################

tlist = np.linspace(0.0, 100.0, 200)

# constants
w = 1.0  # natural units, w = 1, hbar =1, kb = 1, GHz freq
g = 0.1
N = 30  # num Fock states
n = 0  # photon number
# Bases
basis_atom_e = q.basis(2, 0)
basis_qho = q.basis(N, n)


# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()


psi0 = q.tensor(basis_atom_e, basis_qho)

# Jaynes-Cummings Hamiltonian
H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)


############################ SIMULATION ###################################
e_ops = [adag * a, s_raise * s_lower]

sim = TLSQHOSimulator(H, psi0, e_ops=e_ops, times=tlist)
results = sim.evolve()
results_expt = sim.expect(results)
vne = sim.vne(results)
coh_tls = sim.rel_coherence(results, subsys="TLS")
coh_qho = sim.rel_coherence(results, subsys="QHO")


############################### PLOTS #####################################
sim.plot(
    "CQS_expt",
    [
        {
            "y_data": results_expt[0],
            "label": "Atom excited populations",
            "colour": "tab:blue",
        },
        {
            "y_data": results_expt[1],
            "label": "Cavity photon number",
            "colour": "tab:red",
        },
    ],
    y_label="Expectation Values",
    savepath="JCM",
)

sim.plot(
    "CQS_vne",
    [{"y_data": vne}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="JCM",
)

sim.plot(
    "CQS_coh",
    [
        {"y_data": coh_tls, "label": "Atomic subsystem", "colour": "tab:blue"},
        {"y_data": coh_qho, "label": "Cavity subsystem", "colour": "tab:red"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="JCM",
)
