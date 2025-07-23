from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math


############################## SETUP ######################################

# NCOMMS4012: Consider pigments-protein environment of PE545 complex. Taking parameters stated,
# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = w*(0.0578)^0.5 = 267 cm^-1
# Moreover, T ~ 300K room temp expt.

t_max = 5.0  # ps
tlist = np.linspace(0.0, 2 * np.pi * 0.03 * t_max, 300)
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
basis_atom_g = q.basis(2, 1)
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

neg_tls_spont = sim_open_spont.negativity(results_open_spont, subsys="TLS")
neg_qho_spont = sim_open_spont.negativity(results_open_spont, subsys="QHO")

coh_tls_spont = sim_open_spont.rel_coherence(results_open_spont, subsys="TLS")
coh_qho_spont = sim_open_spont.rel_coherence(results_open_spont, subsys="QHO")

# Thermal Dissipation
sim_open_therm = TLSQHOSimulator(H, psi0, L_therm, e_ops, times=tlist)
results_open_therm = sim_open_therm.evolve()

expt_open_therm = sim_open_therm.expect(results_open_therm)

neg_tls_therm = sim_open_therm.negativity(results_open_therm, subsys="TLS")
neg_qho_therm = sim_open_therm.negativity(results_open_therm, subsys="QHO")

coh_tls_therm = sim_open_therm.rel_coherence(results_open_therm, subsys="TLS")
coh_qho_therm = sim_open_therm.rel_coherence(results_open_therm, subsys="QHO")

# Spontaneous Emission + Thermal Dissipation
L_both = L_spont + L_therm
sim_open_both = TLSQHOSimulator(H, psi0, L_both, e_ops, times=tlist)
results_open_both = sim_open_both.evolve()

expt_open_both = sim_open_both.expect(results_open_both)

neg_tls_both = sim_open_both.negativity(results_open_both, subsys="TLS")
neg_qho_both = sim_open_both.negativity(results_open_both, subsys="QHO")

coh_tls_both = sim_open_both.rel_coherence(results_open_both, subsys="TLS")
coh_qho_both = sim_open_both.rel_coherence(results_open_both, subsys="QHO")

############################### PLOTS #####################################

# Negativity

sim_open_spont.plot(
    "OQS_Neg_Spont",
    [
        {
            "y_data": neg_tls_spont,
            "label": "Exciton subsystem",
            "colour": "tab:red",
            "linestyle": "--",
        },
        {
            "y_data": neg_qho_spont,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
            "linestyle": ":",
        },
    ],
    "Negativity",
    savepath="ExVib",
)

sim_open_therm.plot(
    "OQS_Neg_Therm",
    [
        {
            "y_data": neg_tls_therm,
            "label": "Exciton subsystem",
            "colour": "tab:red",
            "linestyle": "--",
        },
        {
            "y_data": neg_qho_therm,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
            "linestyle": ":",
        },
    ],
    "Negativity",
    savepath="ExVib",
)

sim_open_both.plot(
    "OQS_Neg_Both",
    [
        {
            "y_data": neg_tls_both,
            "label": "Exciton subsystem",
            "colour": "tab:red",
            "linestyle": "--",
        },
        {
            "y_data": neg_qho_both,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
            "linestyle": ":",
        },
    ],
    "Negativity",
    savepath="ExVib",
)

# Coherence

sim_open_spont.plot(
    "OQS_Coh_Spont",
    [
        {
            "y_data": coh_tls_spont,
            "label": "Exciton subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_spont,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="ExVib",
)

sim_open_therm.plot(
    "OQS_Coh_Therm",
    [
        {
            "y_data": coh_tls_therm,
            "label": "Exciton subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_therm,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="ExVib",
)

sim_open_both.plot(
    "OQS_Coh_Both",
    [
        {
            "y_data": coh_tls_both,
            "label": "Exciton subsystem",
            "colour": "tab:red",
        },
        {
            "y_data": coh_qho_both,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
        },
    ],
    "Coherence",
    savepath="ExVib",
)

# Populations

sim_open_spont.plot(
    "OQS_Pop_Spont",
    [
        {"y_data": expt_open_spont[0], "label": "Exciton subsystem", "colour": "tab:red"},
        {"y_data": expt_open_spont[1], "label": "Vibration subsystem", "colour": "tab:blue"},
    ],
    "Populations",
    savepath="ExVib",
)

sim_open_therm.plot(
    "OQS_Pop_Therm",
    [
        {"y_data": expt_open_therm[0], "label": "Exciton subsystem", "colour": "tab:red"},
        {"y_data": expt_open_therm[1], "label": "Vibration subsystem", "colour": "tab:blue"},
    ],
    "Populations",
    savepath="ExVib",
)

sim_open_both.plot(
    "OQS_Pop_Both",
    [
        {"y_data": expt_open_both[0], "label": "Exciton subsystem", "colour": "tab:red"},
        {"y_data": expt_open_both[1], "label": "Vibration subsystem", "colour": "tab:blue"},
    ],
    "Populations",
    savepath="ExVib",
)
