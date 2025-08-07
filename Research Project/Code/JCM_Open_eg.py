from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math

############################## SETUP ######################################
t_max = 1500.0  # time in natural units
tlist = np.linspace(0.0, t_max * 0.658, 1000)  # time in ps conversion done here

# constants
w = 1.0  # natural units, w = 1, hbar =1, meV
g = 0.05  # strong coupling regime, C = 4g^2/ gamma*gamma_th >> 1, AND g << w AND Z = 4g^2/w_aw_c < 0.04
N = 2  # num Fock states

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

# Hamiltonian
H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)

# Init cond
psi0 = q.tensor(basis_atom_e + basis_atom_g, basis_qho_0) / math.sqrt(2)

########################## OPEN SIM SETUP #################################

# Constants
gamma = 0.01
gamma_th = 0.01
kbT = 0.001  # meV
n_omega = 1 / (
    np.exp(w / kbT) - 1
)  # low temp regime so n_omega -> 0, optical/IR regime

# Operators
n1_pop = q.tensor(q.qeye(2), basis_qho_1 * basis_qho_1.dag())  # QHO pop |n=1>
e_ops = [s_raise * s_lower, n1_pop]


L_spont = [np.sqrt(gamma) * s_lower]
L_therm = [
    np.sqrt(gamma_th) * (n_omega + 1) * a,  # photon loss
    np.sqrt(gamma_th) * n_omega * adag,  # photon gain
]

############################ SIMULATION ###################################

# Spontaneous Atomic Emission
sim_open_spont = TLSQHOSimulator(H, psi0, L_spont, e_ops, times=tlist)
results_open_spont = sim_open_spont.evolve()

expt_open_spont = sim_open_spont.expect(results_open_spont)

neg_spont = sim_open_spont.negativity(results_open_spont, subsys="TLS")

coh_tls_spont = sim_open_spont.rel_coherence(results_open_spont, subsys="TLS")
coh_qho_spont = sim_open_spont.rel_coherence(results_open_spont, subsys="QHO")

# Thermal Dissipation
sim_open_therm = TLSQHOSimulator(H, psi0, L_therm, e_ops, times=tlist)
results_open_therm = sim_open_therm.evolve()

expt_open_therm = sim_open_therm.expect(results_open_therm)

neg_therm = sim_open_therm.negativity(results_open_therm, subsys="TLS")

coh_tls_therm = sim_open_therm.rel_coherence(results_open_therm, subsys="TLS")
coh_qho_therm = sim_open_therm.rel_coherence(results_open_therm, subsys="QHO")

# Spontaneous Emission + Thermal Dissipation
L_both = L_spont + L_therm
sim_open_both = TLSQHOSimulator(H, psi0, L_both, e_ops, times=tlist)
results_open_both = sim_open_both.evolve()

expt_open_both = sim_open_both.expect(results_open_both)

neg_both = sim_open_both.negativity(results_open_both, subsys="TLS")

coh_tls_both = sim_open_both.rel_coherence(results_open_both, subsys="TLS")
coh_qho_both = sim_open_both.rel_coherence(results_open_both, subsys="QHO")


############################### PLOTS #####################################
# Negativity

sim_open_spont.plot(
    "OQS_Neg_Spont_eg",
    [
        {
            "y_data": neg_spont,
            "colour": "tab:red",
        },
    ],
    "Negativity",
    savepath="JCM",
)

sim_open_therm.plot(
    "OQS_Neg_Therm_eg",
    [
        {
            "y_data": neg_therm,
            "colour": "tab:red",
        },
    ],
    "Negativity",
    savepath="JCM",
)

sim_open_both.plot(
    "OQS_Neg_Both_eg",
    [
        {
            "y_data": neg_both,
            "colour": "tab:red",
        },
    ],
    "Negativity",
    savepath="JCM",
)

# Coherence

sim_open_spont.plot(
    "OQS_Coh_Spont_eg",
    [
        {
            "y_data": coh_tls_spont,
            "label": "Atomic subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_spont,
            "label": "Cavity subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="JCM",
)

sim_open_therm.plot(
    "OQS_Coh_Therm_eg",
    [
        {
            "y_data": coh_tls_therm,
            "label": "Atomic subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_therm,
            "label": "Cavity subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="JCM",
)

sim_open_both.plot(
    "OQS_Coh_Both_eg",
    [
        {
            "y_data": coh_tls_both,
            "label": "Atomic subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_both,
            "label": "Cavity subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="JCM",
)

# Populations

sim_open_spont.plot(
    "OQS_Pop_Spont_eg",
    [
        {
            "y_data": expt_open_spont[0],
            "label": "Excited Atomic State",
            "colour": "tab:red",
        },
        {
            "y_data": expt_open_spont[1],
            "label": "Cavity photon number, n = 1",
            "colour": "tab:blue",
        },
    ],
    "Populations",
    savepath="JCM",
)

sim_open_therm.plot(
    "OQS_Pop_Therm_eg",
    [
        {
            "y_data": expt_open_therm[0],
            "label": "Excited Atomic State",
            "colour": "tab:red",
        },
        {
            "y_data": expt_open_therm[1],
            "label": "Cavity photon number, n = 1",
            "colour": "tab:blue",
        },
    ],
    "Populations",
    savepath="JCM",
)

sim_open_both.plot(
    "OQS_Pop_Both_eg",
    [
        {
            "y_data": expt_open_both[0],
            "label": "Excited Atomic State",
            "colour": "tab:red",
        },
        {
            "y_data": expt_open_both[1],
            "label": "Cavity photon number, n = 1",
            "colour": "tab:blue",
        },
    ],
    "Populations",
    savepath="JCM",
)
