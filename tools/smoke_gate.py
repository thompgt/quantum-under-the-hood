"""Phase 0 smoke gate.

Verifies that the exact API surface the 30 notebooks depend on actually works on
this machine's Python 3.13 / numpy 2.x / Aer 0.17.2 combination. Aer declares
`qiskit>=1.1.0` with no upper bound, so pip will happily install a pairing nobody
CI-tested; this script is what catches that.

Run:  .venv\\Scripts\\python.exe tools\\smoke_gate.py
Exit code 0 means notebook authoring may begin.
"""

import sys

import numpy as np


def check(label, fn):
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - we want every failure reported, not the first
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"  ok    {label}: {value}")
    return True


def main():
    import qiskit
    import qiskit_aer

    print(f"qiskit {qiskit.__version__} | aer {qiskit_aer.__version__} | "
          f"numpy {np.__version__} | python {sys.version.split()[0]}")

    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler
    from qiskit.quantum_info import (
        DensityMatrix,
        Kraus,
        Operator,
        PTM,
        SparsePauliOp,
        Statevector,
        entropy,
        partial_trace,
        state_fidelity,
    )
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit.visualization import (
        plot_bloch_multivector,
        plot_histogram,
        plot_state_city,
        plot_state_hinton,
        plot_state_qsphere,
    )
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import (
        NoiseModel,
        amplitude_damping_error,
        depolarizing_error,
        thermal_relaxation_error,
    )
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    ok = True

    # --- Bell state, the repo's canonical circuit -------------------------------
    qr, cr = QuantumRegister(2, "q"), ClassicalRegister(2, "c")
    bell_meas = QuantumCircuit(qr, cr)
    bell_meas.h(0)
    bell_meas.cx(0, 1)
    bell_meas.measure([0, 1], [0, 1])

    bell = QuantumCircuit(2)  # no measurements - required for Statevector/Estimator
    bell.h(0)
    bell.cx(0, 1)

    # --- 1. AerSimulator.run + transpile ---------------------------------------
    backend = AerSimulator()
    ok &= check(
        "AerSimulator.run",
        lambda: backend.run(transpile(bell_meas, backend), shots=1024,
                            seed_simulator=42).result().get_counts(),
    )

    # --- 2. Primitives V2 -------------------------------------------------------
    ok &= check(
        "StatevectorSampler",
        lambda: StatevectorSampler(seed=42).run([bell_meas], shots=1024)
        .result()[0].data.c.get_counts(),
    )
    ok &= check(
        "Aer SamplerV2",
        lambda: SamplerV2(seed=42).run([bell_meas], shots=1024)
        .result()[0].data.c.get_counts(),
    )

    obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XX", 1.0)])
    ok &= check(
        "StatevectorEstimator",
        lambda: StatevectorEstimator(seed=1).run([(bell, obs)]).result()[0].data.evs,
    )
    # NOTE the asymmetry: Aer's SamplerV2 accepts seed=, Aer's EstimatorV2 does NOT.
    # EstimatorV2 seeds only via options={"backend_options": {"seed_simulator": N}}.
    ok &= check(
        "Aer EstimatorV2",
        lambda: EstimatorV2(options={"backend_options": {"seed_simulator": 1}})
        .run([(bell, obs)]).result()[0].data.evs,
    )

    # --- 3. quantum_info --------------------------------------------------------
    ok &= check("Statevector", lambda: np.round(Statevector(bell).data, 4))
    ok &= check("Operator", lambda: Operator(bell).data.shape)
    ok &= check("partial_trace + entropy",
                lambda: round(float(entropy(partial_trace(Statevector(bell), [1]))), 6))
    ok &= check("DensityMatrix purity",
                lambda: round(float(DensityMatrix(bell).purity().real), 6))
    ok &= check("state_fidelity",
                lambda: round(state_fidelity(Statevector(bell), Statevector(bell)), 6))
    ok &= check("Kraus -> PTM", lambda: PTM(Kraus(amplitude_damping_error(0.3))).data.shape)

    # --- 4. Noise ---------------------------------------------------------------
    def noisy_counts():
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 1), ["h", "x", "sx", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 2), ["cx"])
        nm.add_all_qubit_quantum_error(thermal_relaxation_error(100e-6, 80e-6, 1e-6), ["id"])
        nb = AerSimulator(noise_model=nm)
        return nb.run(transpile(bell_meas, nb), shots=1024, seed_simulator=7).result().get_counts()

    ok &= check("NoiseModel", noisy_counts)

    # --- 5. Dynamic circuits: if_test (the .c_if replacement, needed by B25) ----
    def dynamic():
        qc = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, "c"))
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.cregs[0], 1)):
            qc.x(1)
        qc.measure_all()
        b = AerSimulator()
        return b.run(transpile(qc, b), shots=256, seed_simulator=3).result().get_counts()

    ok &= check("if_test dynamic circuit", dynamic)

    # --- 6. Transpiler ----------------------------------------------------------
    ok &= check(
        "generate_preset_pass_manager",
        lambda: generate_preset_pass_manager(
            optimization_level=3, backend=backend, seed_transpiler=7).run(bell).depth(),
    )

    # --- 7. Circuit library function forms --------------------------------------
    def library():
        from qiskit.circuit.library import QFTGate, efficient_su2, grover_operator
        return (QFTGate(3).num_qubits,
                efficient_su2(3, reps=2).num_parameters,
                grover_operator(QuantumCircuit(3)).num_qubits)

    ok &= check("circuit.library function forms", library)

    # --- 8. Drawing / visualization (pylatexenc-dependent) ----------------------
    def draw_mpl():
        import matplotlib
        matplotlib.use("Agg")
        fig = bell_meas.draw("mpl")
        return type(fig).__name__

    ok &= check("circuit.draw('mpl')", draw_mpl)

    sv = Statevector(bell)
    for name, fn in [
        ("plot_bloch_multivector", lambda: plot_bloch_multivector(sv)),
        ("plot_state_qsphere", lambda: plot_state_qsphere(sv)),
        ("plot_state_city", lambda: plot_state_city(sv)),
        ("plot_state_hinton", lambda: plot_state_hinton(sv)),
        ("plot_histogram", lambda: plot_histogram({"00": 500, "11": 524})),
    ]:
        ok &= check(name, lambda fn=fn: type(fn()).__name__)

    # --- 9. Endianness contract -------------------------------------------------
    def endianness():
        qc = QuantumCircuit(2)
        qc.x(0)  # set qubit 0 only
        data = Statevector(qc).data
        idx = int(np.argmax(np.abs(data)))
        assert idx == 1, f"expected little-endian index 1 for |01>, got {idx}"
        return f"X on q0 -> index {idx} -> label '{Statevector(qc).draw('latex_source')}'"

    ok &= check("little-endian contract", endianness)

    print()
    if ok:
        print("SMOKE GATE PASSED - notebook authoring may begin.")
        return 0
    print("SMOKE GATE FAILED - do not start authoring.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
