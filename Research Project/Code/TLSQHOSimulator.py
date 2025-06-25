import qutip as q
import numpy as np
from matplotlib import pyplot as plt


class TLSQHOSimulator:
    def __init__(
        self,
        H: q.core.qobj.Qobj,
        psi0: q.core.qobj.Qobj,
        L=None,
        e_ops=None,
        times=None,
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
        if self.e_ops is None:
            raise Exception(
                "Cannot provide expectation values if no projectors are specified."
            )
        return results.expect
    
    def vne(self, results: q.solver.result.Result):
        atom_subsys = [state.ptrace(0) for state in results.states]
        return [q.entropy_vn(dm) for dm in atom_subsys]

    def rel_coherence(self, results: q.solver.result.Result):
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
        savepath = f"results/{filename}"

        all_arrays = all(isinstance(item, np.ndarray) for item in y_data)

        if all_arrays:
            if legend is None:
                raise ValueError("Legend must be provided if plotting multiple lines.")
            if len(legend)!= len(y_data):
                raise ValueError("Length of Legend list must match number of expectation projectors.")
            
            for item, label in zip(y_data, legend):
                plt.plot(self.times, item, label=label)
            plt.legend()
        else:
            plt.plot(self.times, y_data)
            if legend is not None:
                if len(legend) !=1:
                    raise ValueError("Must provide only one legend for single data line plot.")
                plt.legend(legend)
                


        plt.xlabel("Time")
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.savefig(savepath)
        plt.close()
        print(f"Plot successfully saved to {savepath}.")
