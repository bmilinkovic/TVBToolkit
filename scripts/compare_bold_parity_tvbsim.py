"""Developer parity check between TVBToolkit and local TVBSim BOLD helpers.

This script is optional and not used by package runtime/tests. It compares the
ported legacy routines (`butter_filtering`, `corr_FC_SC`) against the local
TVBSim reference file when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tvbtoolkit.bold import BOLDParams, butter_filtering, corr_FC_SC


def _load_tvbsim_namespace() -> dict:
    ref = Path("/Users/borjan/CNRS/projects/TVBSim/tvbsim/BOLD.py")
    if not ref.exists():
        raise FileNotFoundError(f"Reference not found: {ref}")
    src = ref.read_text(encoding="utf-8").replace("from common import create_dicts\n", "")
    ns: dict = {}
    exec(compile(src, str(ref), "exec"), ns)  # noqa: S102
    return ns


def main() -> None:
    ns = _load_tvbsim_namespace()

    rng = np.random.default_rng(123)
    x = rng.normal(size=(2000, 16))
    bp = BOLDParams(TR=2.0, n_order=2, low_f_num=0.01, high_f_num=0.1)

    y_new = butter_filtering(x, bp)
    y_ref = ns["butter_filtering"](x, ns["BOLDParams"](TR=2.0, n_order=2, low_f_num=0.01, high_f_num=0.1))
    print("butter_filtering max abs diff:", float(np.max(np.abs(y_new - y_ref))))

    sig = rng.normal(size=(16, 800))
    a = rng.uniform(size=(16, 16))
    sc = (a + a.T) / 2
    np.fill_diagonal(sc, 0.0)

    fc_new, c_new = corr_FC_SC(sig, sc)
    fc_ref, c_ref = ns["corr_FC_SC"](sig, sc)

    print("corr_FC_SC max abs diff (FC):", float(np.max(np.abs(fc_new - fc_ref))))
    print("corr_FC_SC abs diff (coef):", float(abs(c_new - c_ref)))


if __name__ == "__main__":
    main()
