import qutip as q
import numpy as np
from matplotlib import pyplot as plt


class TLSQHOSimulator:
    """
    This class streamlines QuTip simulations by allowing the user to input a Hamiltonian and an initial state,
    and evolve the state via either Open or Closed Quantum System approaches. Moreover, the user may choose to
    plot and save the results within the repository itself.

    Instance attributes:
        H(q.core.qobj.Qobj): Hamiltonian which governs evolution dynamics.
        psi0(q.core.qobj.Qobj): Initial state at t=0 from which the system evolves.
        L(q.core.qobj.Qobj): Optional, Jump operator which governs decay of Open Quantum Systems.
        e_ops(list(q.core.qobj.Qobj)): Optional, list of projectors to calculate expectation values from.
        times(np.ndarray): Optional, numpy array of times for which to evolve the initial state from.


    Class methods:
        evolve(self): Evolves the state given at minimum a Hamiltonian and initial state, using QuTip's mesolve().
        expect(self, results): Calculates expectation values of results if at least one e_ops is provided.
        vne(self, results): Calculates Von Neumann Entropy for a subsystem of an evolved state.
        rel_coherence(self, results): Calculates Relative Entropy of Coherence for a subsystem of an evolved state.
        plot(self, y_data, title, y_label, filename, legend): Plots data and saves to specific location in repository.

    """

    def __init__(
        self,
        H: q.core.qobj.Qobj,
        psi0: q.core.qobj.Qobj,
        L: q.core.qobj.Qobj = None,
        e_ops: list = None,
        times: np.ndarray = None,
    ):
        self.H = H
        self.psi0 = psi0

        if L is None:
            self.L = []
        else:
            self.L = L

        if times is None:
            self.times = np.linspace(0.0, 20.0, 200)
        else:
            self.times = times

        if e_ops is None:
            self.e_ops = []
        else:
            self.e_ops = e_ops

    def evolve(self):
        """
        Evolves the initial state via given Hamiltonian using the mesolve() function provided by QuTip.
        """
        results = q.mesolve(
            self.H,
            self.psi0,
            self.times,
            self.L,
            self.e_ops,
            options=q.Options(store_states=True),
        )

        return results

    def expect(self, results: q.solver.result.Result):
        """
        Calculates expectation values of results if at least one e_ops is provided.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.

        """
        if self.e_ops is None:
            raise Exception(
                "Cannot provide expectation values if no projectors are specified."
            )
        return results.expect

    def vne(self, results: q.solver.result.Result):
        """
         Calculates Von Neumann Entropy for a subsystem of an evolved state.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.
        """
        atom_subsys = [state.ptrace(0) for state in results.states]
        return [q.entropy_vn(dm) for dm in atom_subsys]

    def rel_coherence(self, results: q.solver.result.Result):
        """
        Calculates Relative Entropy of Coherence for a subsystem of an evolved state.

        Args:
            results(q.solver.result.Result): Output of evolve() method, which is a solver type for QuTip.

        """
        coherence = []

        for ket in results.states:
            if ket.isket:
                dm = q.ket2dm(ket)
            else:
                dm = ket

            # density matrix of diagonal elements
            diag_element = dm.diag()
            dm_diag = q.Qobj(np.diag(diag_element), dims=dm.dims)

            S_dm_diag = q.entropy_vn(dm_diag)
            S_dm = q.entropy_vn(dm)

            # applying relative entropy of coherence equation
            coherence_rel_ent = S_dm_diag - S_dm
            coherence.append(coherence_rel_ent)

        return coherence

    def plot(
        self, y_data: list, title: str, y_label: str, filename: str, legend: list = None
    ):
        """
        Plots data and saves to specific location in repository.

        Args:
            y_data(list) : A list/array of data which the user plots against times. It should be the outputs 
            of vne(), rel_coherence() and expect() methods of this class.
            title(str) : Name of graph.
            y_label(str) : Y-axis label.
            filename(str) : Name of file which will be saved.
            legend(list) : A list of strings which must be provided if at least one expectation operator is 
            present and the user implements the expect() method. 
                        

        """
        savepath = f"results/{filename}"

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
