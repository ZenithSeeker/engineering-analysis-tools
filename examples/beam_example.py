"""Simply supported beam with a midspan point load and a UDL, solved with Macaulay's method."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beam_analysis import Beam, BeamSolver, PointLoad, Support, UDL

beam = Beam(
    length=10,
    supports=[Support(position=0, type="pin"), Support(position=10, type="roller")],
    loads=[PointLoad(position=5, magnitude=-10), UDL(start=0, end=10, magnitude=-2)],
)
solver = BeamSolver(beam)
solver.solve()
solver.print_results()
solver.plot()
plt.savefig("docs/beam_example.png", dpi=150, bbox_inches="tight")
