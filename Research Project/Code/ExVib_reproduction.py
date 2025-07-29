import qutip as q
import numpy as np
import math
from matplotlib import pyplot as plt

############################## SETUP ########################################
t_max = 1.0  # ps
tlist = np.linspace(0.0, 2 * np.pi * 0.03 * t_max, 300)

# params, cm^-1
w = 1111
d_epsilon = 1042
V = 92
g = w * math.sqrt(0.0578)
N = 20  # num Fock states
T = 300
kbT = 0.695 * T
n_omega = 1 / (np.exp(w / kbT) - 1)  # Expectation value for number of particles in thermal state

# Bases
basis_atom_e = q.basis(2, 0)
basis_atom_g = q.basis(2, 1)
basis_qho_0 = q.basis(N, 0) 

# Operators
a = q.tensor(q.qeye(2), q.destroy(N))
adag = a.dag()
s_z = q.tensor(q.sigmaz(), q.qeye(N))
s_x = q.tensor(q.sigmax(), q.qeye(N))
s_lower = q.tensor(q.sigmam(), q.qeye(N))
s_raise = s_lower.dag()

# Exciton-Vibration Hamiltonian
H_tls = 0.5 * d_epsilon * q.sigmaz() + V * q.sigmax()
H_ex = q.tensor(H_tls, q.qeye(N))
H_vib = w * adag * a
H_int = -(g / math.sqrt(2)) * q.tensor(q.sigmaz(), q.destroy(N).dag() + q.destroy(N))
H = H_ex + H_vib + H_int


#################### TLS FREE HAMILTONIAN EIGENBASIS ###############################

eigvals, eigvecs = H_tls.eigenstates()
Y = eigvecs[0]
X = eigvecs[1]
Y0 = q.tensor(Y, basis_qho_0)
X0 = q.tensor(X, basis_qho_0)

############################## INTIAL STATE ########################################

# Write as rho0 since thermal_dm is a density matrix, |X,0>
rho0 = q.tensor(X * X.dag(), q.thermal_dm(N, n_omega))


############################# e_ops #################################################

e_ops = [q.tensor(Y * Y.dag(), q.qeye(N))]  # pop of ground state

############################# SIMULATION ###########################################

result = q.mesolve(H, rho0, tlist, e_ops=e_ops, options=q.Options(store_states=True))

pop_Y = result.expect[0]  # list of lists [[]] so must be extracted this way

# Coherence
coherence = []
for state in result.states:
    coh = X0.dag() * state * Y0
    coherence.append(abs(coh))


plt.figure(figsize=(8, 5))
plt.plot(tlist, pop_Y, linewidth=2, color="blue")
plt.plot(tlist, coherence, linewidth=0.7, color="tab:red")
plt.xlabel("t (ps)")
plt.ylim(0)
plt.xlim(0)
plt.tight_layout()
plt.savefig("results/ExVib/fig2b.png")