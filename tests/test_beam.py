"""Check the Macaulay solver against standard closed-form beam results (EI = 1)."""

import pytest

from beam_analysis import Beam, BeamSolver, PointLoad, PointMoment, Support, UDL, x


def solve(beam):
    s = BeamSolver(beam)
    s.solve()
    return s


def at(expr, pos):
    return float(expr.subs(x, pos))


def test_simply_supported_udl():
    L, w = 10.0, -2.0
    s = solve(Beam(L, [Support(0, "pin"), Support(L, "roller")], [UDL(0, L, w)]))
    assert at(s.V_solved, 1e-9) == pytest.approx(-w * L / 2)            # left reaction appears in shear
    assert at(s.M_solved, L / 2) == pytest.approx(-w * L ** 2 / 8)       # wL^2/8 sagging
    assert at(s.y_solved, L / 2) == pytest.approx(5 * w * L ** 4 / 384)  # 5wL^4/384EI, downward


def test_simply_supported_midspan_point_load():
    L, P = 6.0, -12.0
    s = solve(Beam(L, [Support(0, "pin"), Support(L, "roller")], [PointLoad(L / 2, P)]))
    assert at(s.M_solved, L / 2) == pytest.approx(-P * L / 4)
    assert at(s.y_solved, L / 2) == pytest.approx(P * L ** 3 / 48)


def test_cantilever_tip_load():
    L, P = 3.0, -5.0
    s = solve(Beam(L, [Support(0, "fixed")], [PointLoad(L, P)]))
    assert at(s.M_solved, 1e-9) == pytest.approx(P * L)                  # hogging at the wall
    assert at(s.y_solved, L) == pytest.approx(P * L ** 3 / 3)
    assert at(s.theta_solved, L) == pytest.approx(P * L ** 2 / 2)


def test_cantilever_udl():
    L, w = 4.0, -1.5
    s = solve(Beam(L, [Support(0, "fixed")], [UDL(0, L, w)]))
    assert at(s.y_solved, L) == pytest.approx(w * L ** 4 / 8)


def test_cantilever_end_moment():
    L, M0 = 2.0, 7.0
    s = solve(Beam(L, [Support(0, "fixed")], [PointMoment(L, M0)]))
    # a CCW moment at the free end bends the tip upward
    assert at(s.y_solved, L) == pytest.approx(M0 * L ** 2 / 2)


def test_load_at_origin_is_not_dropped():
    # an overhanging beam with a point load right at x = 0
    s = solve(Beam(10.0, [Support(2, "pin"), Support(10, "roller")], [PointLoad(0, -4.0)]))
    assert at(s.V_solved, 1e-9) == pytest.approx(-4.0)
