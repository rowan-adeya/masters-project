from TLSQHOSimulator import TLSQHOSimulator
import qutip as q
import numpy as np
import math

"""
This simulation looks at the Exciton Vibration system and plots the negativity of each subsystem
for 3 Lindblad Decay Operators: Spontaneous Atomic Emission, Thermal Dissipation, and both.

"""
############################## SETUP ######################################

# d_epsilon = 1042, V = 92, d_E = 1058, w = 1111 all in cm^-1.
# g = w*(0.0578)^0.5 = 267 cm^-1
# Moreover, T ~ 300K room temp expt.

t_max = 100.0  # ps
tlist = np.linspace(0.0, 2 * np.pi * 0.03 * t_max, 300)

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
basis_qho_0 = q.basis(N, 0)  # 0 photons in vibration initially
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

######################## INITIAL CONDITIONS ###############################
initial_states = {
    "e0": q.tensor(basis_atom_e, basis_qho_0),
    "e1": q.tensor(basis_atom_e, basis_qho_1),
    "g0": q.tensor(basis_atom_g, basis_qho_0),
    "g1": q.tensor(basis_atom_g, basis_qho_1),
}
########################## OPEN SIM SETUP #################################

# Constants
gamma = 1.0
gamma_th = 1.0
T = 300
kbT = 0.695 * T
n_omega = 1 / (np.exp(w / kbT) - 1)

# Operators
e_ops = [adag * a, s_raise * s_lower, s_lower * s_raise]

L_spont = [np.sqrt(gamma) * s_lower]
L_therm = [
    np.sqrt(gamma_th) * (n_omega + 1) * a,  # photon loss
    np.sqrt(gamma_th) * n_omega * adag,  # photon gain
]
L_both = L_spont + L_therm

decay_ops = {
    "spont": L_spont,
    "therm": L_therm,
    "both": L_both,
}
############################ SIMULATION ###################################

for decay_label, lindblad_ops in decay_ops.items():
    neg_data = []
    for psi_label, psi in initial_states.items():
        sim = TLSQHOSimulator(H, psi, lindblad_ops, e_ops, times=tlist)
        results = sim.evolve()
        neg = sim.negativity(results)
        neg_data.append({
            "y_data": neg,
            "label": f"|{psi_label[0]},{psi_label[1]}>",
            "colour": {
                "e0": "tab:red",
                "e1": "tab:blue",
                "g0": "tab:green",
                "g1": "tab:orange",
            }[psi_label],
        })
    sim.plot(
        f"neg_{decay_label}",
        neg_data,
        "Negativity",
        savepath="ExVib/Open/Negativity",
    )
