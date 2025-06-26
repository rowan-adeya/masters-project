import qutip as q
import numpy as np
import pytest
from pathlib import Path
from TLSQHOSimulator import TLSQHOSimulator


@pytest.fixture
def setup_simulator():
    N = 2
    w = 1.0
    g = 1.0

    basis_atom_e = q.basis(2, 0)
    basis_qho_0 = q.basis(N, 0)
    psi0 = q.tensor(basis_atom_e, basis_qho_0)

    a = q.tensor(q.qeye(2), q.destroy(N))
    adag = a.dag()
    s_z = q.tensor(q.sigmaz(), q.qeye(N))
    s_lower = q.tensor(q.sigmam(), q.qeye(N))
    s_raise = s_lower.dag()

    psi0 = q.tensor(basis_atom_e, basis_qho_0)
    H = 0.5 * w * s_z + w * (adag * a + 0.5) + g * (adag * s_lower + a * s_raise)
    L = 1.0 * s_lower
    e_ops = adag * a

    sim = TLSQHOSimulator(H, psi0, L, e_ops)
    return sim


def test_init_good(setup_simulator):
    assert setup_simulator.psi0.isket
    assert setup_simulator.H.isoper
    assert all(op.isoper for op in setup_simulator.L)
    assert all(op.isoper for op in setup_simulator.e_ops)
    assert isinstance(setup_simulator.L, list)
    assert isinstance(setup_simulator.e_ops, list)
    np.testing.assert_allclose(setup_simulator.times, np.linspace(0.0, 20.0, 200))

def test_init_raises(setup_simulator):

    with pytest.raises(TypeError, match="The Hamiltonian must be a QuTip Qobj of type 'oper'."):
        H = q.basis(2,1)
        psi0 = q.basis(2,1)
        sim = TLSQHOSimulator(H, psi0)

    with pytest.raises(TypeError, match="The initial state must be a QuTip Qobj of type 'ket'."):
        H = setup_simulator.H
        psi0 = q.basis(2,1).dag()
        sim = TLSQHOSimulator(H, psi0)

    with pytest.raises(TypeError, match="The jump operator must be a QuTip Qobj of type 'oper'."):
        H = setup_simulator.H
        psi0 = setup_simulator.psi0
        L = 2
        sim = TLSQHOSimulator(H, psi0, L)

    with pytest.raises(TypeError, match="Each expectation operator must be a QuTip Qobj of type 'oper'."):
        H = setup_simulator.H
        psi0 = setup_simulator.psi0
        L = setup_simulator.L
        e_ops = "Hello"
        sim = TLSQHOSimulator(H, psi0, L, e_ops)

    with pytest.raises(TypeError, match="The list of times must be a NumPy array."):
        H = setup_simulator.H
        psi0 = setup_simulator.psi0
        L = setup_simulator.L
        e_ops = setup_simulator.e_ops
        times = 0
        sim = TLSQHOSimulator(H, psi0, L, e_ops, times)



def test_vne_output(setup_simulator):
    sim = setup_simulator
    result = sim.evolve()
    vne_list = sim.vne(result)

    assert isinstance(vne_list, list)
    assert len(vne_list) == len(sim.times)
    assert all(isinstance(val, float) for val in vne_list)


def test_rel_coherence_output(setup_simulator):
    sim = setup_simulator
    result = sim.evolve()
    coherence_list = sim.rel_coherence(result)

    assert isinstance(coherence_list, list)
    assert len(coherence_list) == len(sim.times)
    assert all(isinstance(val, float) for val in coherence_list)


def test_expect_output(setup_simulator):
    sim = setup_simulator
    result = sim.evolve()
    expect_vals = sim.expect(result)

    assert isinstance(expect_vals, list)
    assert len(expect_vals[0]) == len(sim.times)


def test_plot_function(setup_simulator):
    sim = setup_simulator
    result = sim.evolve()
    vne_vals = sim.vne(result)

    # File will be saved in results/ according to simulator logic
    filename = "test_plot.png"
    filepath = Path("results") / filename

    sim.plot(vne_vals, title="Test", y_label="Entropy", filename=filename)

    assert filepath.exists()

    # Clean up
    filepath.unlink()