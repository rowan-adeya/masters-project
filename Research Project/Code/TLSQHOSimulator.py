import qutip as q
import numpy as np
from matplotlib import pyplot as plt


class TLSQHOSimulator:
    """
    This class streamlines QuTip simulations by allowing the user to input a Hamiltonian and an initial state,
    and evolve the state via either Open or Closed Quantum System approaches. Moreover, the user may choose to
    plot and save the results within the repository itself.

    Instance attributes:
        H(q.Qobj): Hamiltonian which governs evolution dynamics.
        psi0(q.Qobj): Initial state at t=0 from which the system evolves.
        L(q.Qobj): Optional, Jump operator which governs decay of Open Quantum Systems.
        e_ops(list(q.Qobj)): Optional, list of projectors to calculate expectation values from.
        times(np.ndarray): Optional, numpy array of times for which to evolve the initial state from.

    Class methods:
        evolve(): Evolves the state given at minimum a Hamiltonian and initial state, using QuTip's mesolve().
        expect(results): Calculates expectation values of results if at least one e_ops is provided.
        vne(results, subsys): Calculates Von Neumann Entropy for a subsystem of an evolved state.
        negativity(results, subsys): Calculates the Negativity for a subsystem of an evolved state.
        rel_coherence(results): Calculates Relative Entropy of Coherence for a subsystem of an evolved state.
        plot(y_data, title, y_label, filename, legend): Plots data and saves to specific location in repository.
    """

    def __init__(
        self,
        H: q.Qobj,
        psi0: q.Qobj,
        L: q.Qobj = None,
        e_ops: list = None,
        times: np.ndarray = None,
    ):
        if not (H.type == "oper" and isinstance(H, q.Qobj)):
            raise TypeError("The Hamiltonian must be a QuTip Qobj of type 'oper'.")
        self.H = H

        if not (psi0.type == "ket" and isinstance(psi0, q.Qobj)):
            raise TypeError("The initial state must be a QuTip Qobj of type 'ket'.")
        self.psi0 = psi0

        if L is None:
            self.L = []
        else:
            if not isinstance(L, list):
                L = [L]
            for op in L:
                if not isinstance(op, q.Qobj) or op.type != "oper":
                    raise TypeError(
                        "The jump operator must be a QuTip Qobj of type 'oper'."
                    )
            self.L = L

        if e_ops is None:
            self.e_ops = []
        else:
            if not isinstance(e_ops, list):
                e_ops = [e_ops]
            for op in e_ops:
                if not isinstance(op, q.Qobj) or op.type != "oper":
                    raise TypeError(
                        "Each expectation operator must be a QuTip Qobj of type 'oper'."
                    )
            self.e_ops = e_ops

        if times is None:
            self.times = np.linspace(
                0.0, 1000.0, 200
            )  # if using natural units, its in units of 1/w which if w = 1GHz is 1ns
        else:
            if not isinstance(times, np.ndarray):
                raise TypeError("The list of times must be a NumPy array.")
            self.times = times

    def evolve(self):
        """
        Calculates the result of the time evolution using the mesolve() function provided by QuTip.

        Returns:
            results(q.solver.result.Result): A solver type which can be acted upon to extract evolved states.
        """
        results = q.mesolve(
            self.H,
            self.psi0,
            self.times,
            self.L,
            self.e_ops,
            options=q.Options(store_states=True),
        )
        if not results.states:
            raise ValueError(
                "No states were found in the results object after evolution."
            )
        else:
            return results

    def expect(self, results: q.solver.result.Result):
        """
        Calculates expectation values of results if at least one e_ops is provided.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.
        Returns:
            results.expect(list): List of expectation values for the given expectation operators.
        """
        if not self.e_ops:
            raise ValueError(
                "Cannot provide expectation values if no projectors are specified."
            )

        return results.expect

    def vne(self, results: q.solver.result.Result, subsys: int = 0):
        """
        Calculates Von Neumann Entropy for a subsystem of an evolved state. Not applicable to Open Quantum
        Systems.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.
            subsys(int): Default = 0, which subsystem to partial trace over.
        Returns:
            vne(list): The Von Neumann Entropy of each density matrix in the result.states object.
        """
        if subsys not in (0, 1):
            raise ValueError("subsys must be either 0 or 1.")

        atom_subsys = [state.ptrace(subsys) for state in results.states]
        vne = [q.entropy_vn(dm) for dm in atom_subsys]
        return vne

    def negativity(self, results: q.solver.result.Result, subsys: str = "TLS"):
        """
        Calculates the Negativity for a subsystem of an evolved state. The negativity is the absolute
        sum of the negative eigenvalues. The partial transpose function takes an argument, mask, is a
        list of bools (0 or 1), where a 1 in the list tells the function to perform the partial
        transpose.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.
            subsys(str): Default = TLS, which subsystem to calculate Negativity for.

        Returns:
            neg_vals(list): The negativity of each density matrix in the result.states object.
        """
        allowed_subsys_vals = ("TLS", "QHO")
        if subsys not in allowed_subsys_vals:
            raise ValueError(f"subsys must be either {allowed_subsys_vals}.")

        neg_vals = []

        for state in results.states:
            if state.isket:
                rho = q.ket2dm(state)
            else:
                rho = state

            if subsys == "TLS":
                mask = [0, 1]  # mask tells which subsys to perform partial transpose on
            else:
                mask = [1, 0]

            rho_pt = q.partial_transpose(rho, mask)

            eigenvals = rho_pt.eigenenergies()
            negativity = sum(abs(eig) for eig in eigenvals if eig < 0.0)
            neg_vals.append(negativity)

        return neg_vals

    def rel_coherence(self, results: q.solver.result.Result, subsys: str = None):
        """
        Calculates Relative Entropy of Coherence for a subsystem of an evolved state.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.
            subsys(str): Default = None, if not passed, will apply rel_coherence to total system. Otherwise,
            calculate coherence for the given subsystem.
        Returns:
            coherence(list): The coherence of each density matrix in the result.states object.
        """
        if results.states == []:
            raise ValueError("No states were found in the results object.")

        allowed_subsys_vals = (None, "TLS", "QHO")

        if subsys not in allowed_subsys_vals:
            raise ValueError(f"subsys must be an item from {allowed_subsys_vals}.")

        coherence = []

        for state in results.states:
            if state.isket:
                dm = q.ket2dm(state)
            else:
                dm = state

            if subsys == "TLS":
                dm = q.partial_trace(dm, [1])

            elif subsys == "QHO":
                dm = q.partial_trace(dm, [0])

            diag_elements = dm.diag()
            dm_diag = q.Qobj(np.diag(diag_elements), dims=dm.dims)

            S_dm_diag = q.entropy_vn(dm_diag)
            S_dm = q.entropy_vn(dm)
            coherence.append(S_dm_diag - S_dm)

        return coherence

    def plot(
        self,
        y_data: list,
        title: str,
        y_label: str,
        filename: str,
        savepath: str = None,
        legend: list = None,
    ):  # TODO: make sure can convert into cm
        """
        Plots data and saves to specific location in repository.

        Args:
            y_data(list) : A list/array of data which the user plots against times. It should be the outputs
            of vne(), rel_coherence() and expect() methods of this class.
            title(str) : Name of graph.
            y_label(str) : Y-axis label.
            filename(str) : Name of file which will be saved.
            Savepath(str) : Optional, path from "results" directory which the user wants to save their plots to.
            legend(list) : A list of strings which must be provided if at least one expectation operator is
            present and the user implements the expect() method.


        """
        if savepath is None:
            savepath = f"results/{filename}"
        else:
            savepath = f"results/{savepath}/{filename}"

        all_arrays = all(isinstance(item, np.ndarray) for item in y_data)

        if all_arrays:
            if legend is None:
                raise ValueError("Legend must be provided if plotting multiple lines.")
            if len(legend) != len(y_data):
                raise ValueError(
                    "Length of Legend list must match number of expectation projectors."
                )

            for item, label in zip(y_data, legend):
                plt.plot(self.times, item, label=label)
            plt.legend()
        else:
            plt.plot(self.times, y_data)
            if legend is not None:
                if len(legend) != 1:
                    raise ValueError(
                        "Must provide only one legend for single data line plot."
                    )
                plt.legend(legend)

        plt.xlabel("Time")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.savefig(savepath)
        plt.close()
        print(f"Plot successfully saved to {savepath}.")
