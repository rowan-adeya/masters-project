from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math


############################## SETUP ######################################

# NCOMMS4012: Consider pigments-protein environment of PE545 complex. Taking parameters stated,
# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = (0.0578)^0.5 = 267 cm^-1
# Moreover, T ~ 300K room temp expt.
t_max_env = 5
tlist_env = np.linspace(0.0, 2 * np.pi * 0.03 * t_max_env, 2500)
t_max_fast = 0.9  
tlist_fast = np.linspace(0.0, 2 * np.pi * 0.03 * t_max_fast, 2500)

# t in cm conversion, noting original time is in ps

# params, cm^-1
w = 1111
d_epsilon = 1042
V = 92
g = w * math.sqrt(0.0578)
# weak regime, g < gamma, gamma_th AND g << w (zeta = 0.01, C = 1)
N = 10  # num Fock states

# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho_0 = q.basis(N, 0)
basis_qho_1 = q.basis(N, 1)


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
psi0_e0 = q.tensor(basis_atom_e, basis_qho_0)
psi0_eg = q.tensor(basis_atom_e + basis_atom_g, basis_qho_0) / math.sqrt(2)
############################ SIMULATION ENV ###################################
n0_pop = q.tensor(q.qeye(2), basis_qho_0 * basis_qho_0.dag())  # QHO pop |n=0>
e_ops = [s_raise * s_lower, n0_pop]

sim_e0_env = TLSQHOSimulator(H, psi0_e0, e_ops=e_ops, times=tlist_env)
sim_eg_env = TLSQHOSimulator(H, psi0_eg, e_ops=e_ops, times=tlist_env)
results_e0_env = sim_e0_env.evolve()
results_eg_env = sim_eg_env.evolve()


results_expt_e0_env = sim_e0_env.expect(results_e0_env)
results_expt_eg_env = sim_eg_env.expect(results_eg_env)

vne_e0_env = sim_e0_env.vne(results_e0_env)
vne_eg_env = sim_eg_env.vne(results_eg_env)

coh_tls_e0_env = sim_e0_env.rel_coherence(results_e0_env, subsys="TLS")
coh_qho_e0_env = sim_e0_env.rel_coherence(results_e0_env, subsys="QHO")
coh_tot_e0_env = sim_e0_env.rel_coherence(results_e0_env)

coh_tls_eg_env = sim_eg_env.rel_coherence(results_eg_env, subsys="TLS")
coh_qho_eg_env = sim_eg_env.rel_coherence(results_eg_env, subsys="QHO")
coh_tot_eg_env = sim_eg_env.rel_coherence(results_eg_env)


############################ SIMULATION FAST ###################################


sim_e0_fast = TLSQHOSimulator(H, psi0_e0, e_ops=e_ops, times=tlist_fast)
sim_eg_fast = TLSQHOSimulator(H, psi0_eg, e_ops=e_ops, times=tlist_fast)
results_e0_fast = sim_e0_fast.evolve()
results_eg_fast = sim_eg_fast.evolve()

results_expt_e0_fast = sim_e0_fast.expect(results_e0_fast)
results_expt_eg_fast = sim_eg_fast.expect(results_eg_fast)

vne_e0_fast = sim_e0_fast.vne(results_e0_fast)

vne_eg_fast = sim_eg_fast.vne(results_eg_fast)

coh_tls_e0_fast = sim_e0_fast.rel_coherence(results_e0_fast, subsys="TLS")
coh_qho_e0_fast = sim_e0_fast.rel_coherence(results_e0_fast, subsys="QHO")
coh_tot_e0_fast = sim_e0_fast.rel_coherence(results_e0_fast)

coh_tls_eg_fast = sim_eg_fast.rel_coherence(results_eg_fast, subsys="TLS")
coh_qho_eg_fast = sim_eg_fast.rel_coherence(results_eg_fast, subsys="QHO")
coh_tot_eg_fast = sim_eg_fast.rel_coherence(results_eg_fast)

############################### PLOTS ENV #####################################

sim_e0_env.plot(
    "pops_excited",
    [
        {
            "y_data": results_expt_e0_env[0],
            "label": "Exciton excited state",
            "colour": "tab:red",
        },
        {
            "y_data": results_expt_e0_env[1],
            "label": "Vibration photon number, n = 0",
            "colour": "tab:blue",
        },
    ],
    y_label="Expectation Values",
    savepath="/ExVib/Closed/Envelope",
    smooth=150,
)

sim_e0_env.plot(
    "vne",
    [{"y_data": vne_e0_env, "colour": "tab:purple"}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="/ExVib/Closed/Envelope",
    smooth=150
)

sim_e0_env.plot(
    "coh",
    [
        {"y_data": coh_tls_e0_env, "label": "Exciton subsystem", "colour": "tab:red"},
        {"y_data": coh_qho_e0_env, "label": "Vibration subsystem", "colour": "tab:blue"},
        {"y_data": coh_tot_e0_env, "label": "Total system", "colour": "tab:green"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="/ExVib/Closed/Envelope",
    smooth=150
)


sim_eg_env.plot(
    "pops_excited_eg",
    [
        {
            "y_data": results_expt_eg_env[0],
            "label": "Exciton excited state",
            "colour": "tab:red",
        },
        {
            "y_data": results_expt_eg_env[1],
            "label": "Vibration photon number, n = 0",
            "colour": "tab:blue",
        },
    ],
    y_label="Expectation Values",
    savepath="/ExVib/Closed/Envelope",
    smooth=150
)

sim_eg_env.plot(
    "vne_eg",
    [{"y_data": vne_eg_env, "colour": "tab:purple"}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="/ExVib/Closed/Envelope",
    smooth=150
)

sim_eg_env.plot(
    "coh_eg",
    [
        {"y_data": coh_tls_eg_env, "label": "Exciton subsystem", "colour": "tab:red"},
        {"y_data": coh_qho_eg_env, "label": "Vibration subsystem", "colour": "tab:blue"},
        {"y_data": coh_tot_eg_env, "label": "Total system", "colour": "tab:green"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="/ExVib/Closed/Envelope",
    smooth=150
)


############################### PLOTS ENV #####################################

sim_e0_fast.plot(
    "pops_excited",
    [
        {
            "y_data": results_expt_e0_fast[0],
            "label": "Exciton excited state",
            "colour": "tab:red",
        },
        {
            "y_data": results_expt_e0_fast[1],
            "label": "Vibration photon number, n = 0",
            "colour": "tab:blue",
        },
    ],
    y_label="Expectation Values",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)

sim_e0_fast.plot(
    "vne",
    [{"y_data": vne_e0_fast, "colour": "tab:purple"}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)

sim_e0_fast.plot(
    "coh",
    [
        {"y_data": coh_tls_e0_fast, "label": "Exciton subsystem", "colour": "tab:red"},
        {
            "y_data": coh_qho_e0_fast,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
        },
        {"y_data": coh_tot_e0_fast, "label": "Total system", "colour": "tab:green"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)


sim_eg_fast.plot(
    "pops_excited_eg",
    [
        {
            "y_data": results_expt_eg_fast[0],
            "label": "Exciton excited state",
            "colour": "tab:red",
        },
        {
            "y_data": results_expt_eg_fast[1],
            "label": "Vibration photon number, n = 0",
            "colour": "tab:blue",
        },
    ],
    y_label="Expectation Values",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)

sim_eg_fast.plot(
    "vne_eg",
    [{"y_data": vne_eg_fast, "colour": "tab:purple"}],
    y_label="Von Neumann Entanglement Entropy",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)

sim_eg_fast.plot(
    "coh_eg",
    [
        {"y_data": coh_tls_eg_fast, "label": "Exciton subsystem", "colour": "tab:red"},
        {
            "y_data": coh_qho_eg_fast,
            "label": "Vibration subsystem",
            "colour": "tab:blue",
        },
        {"y_data": coh_tot_eg_fast, "label": "Total system", "colour": "tab:green"},
    ],
    y_label="Relative Entropy of Coherence",
    savepath="/ExVib/Closed/Fast",
    smooth=150
)
