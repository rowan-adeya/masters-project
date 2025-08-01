from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math

# https://opg.optica.org/josab/fulltext.cfm?uri=josab-41-8-C206&id=553027
# https://ar5iv.labs.arxiv.org/html/2112.08917v1
# links for weak/strong regimes
############################## SETUP ######################################

tlist = np.linspace(0.0, 200.0, 200)

# constants
w = 1.0  # natural units, w = 1, hbar =1, kb = 1, GHz freq
g = 0.1
N = 30  # num Fock states

# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho_0 = q.basis(N, 0)
basis_qho_1 = q.basis(N, 1)

# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()


psi0 = q.tensor(basis_atom_e + basis_atom_g, basis_qho_0) / math.sqrt(2)

# Jaynes-Cummings Hamiltonian
H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)


############################ SIMULATION ###################################
n0_pop = q.tensor(q.qeye(2), basis_qho_0 * basis_qho_0.dag())  # QHO pop |n=0>
e_ops = [s_raise * s_lower, n0_pop]

sim = TLSQHOSimulator(H, psi0, e_ops=e_ops, times=tlist)
results = sim.evolve()
results_expt = sim.expect(results)
vne = sim.vne(results)
coh_tls = sim.rel_coherence(results, subsys="TLS")
coh_qho = sim.rel_coherence(results, subsys="QHO")


############################### PLOTS #####################################

sim.plot(
    "CQS_expt_eg",
    [
        {
            "y_data": results_expt[0],
            "label": "Excited atomic state",
            "colour": "tab:blue",
        },
        {
            "y_data": results_expt[1],
            "label": "Cavity photon number, n = 0",
            "colour": "tab:red",
        },
    ],
    y_label="Populations",
    savepath="JCM",
)

sim.plot(
    "CQS_vne_eg",
    [{"y_data": vne}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="JCM",
)

sim.plot(
    "CQS_coh_eg",
    [
        {"y_data": coh_tls, "label": "Atomic subsystem", "colour": "tab:blue"},
        {"y_data": coh_qho, "label": "Cavity subsystem", "colour": "tab:red"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="JCM",
)
