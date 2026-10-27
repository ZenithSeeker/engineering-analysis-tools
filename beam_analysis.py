#!/usr/bin/env python3
"""
Beam Analysis Tool using Macaulay's Method

Analytical beam solver that produces shear, moment, slope, and deflection
diagrams using singularity functions (Macaulay brackets).

Run this file directly to see example usage and syntax.
"""

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import SingularityFunction, Symbol, Piecewise, integrate, solve, simplify, Eq


# =============================================================================
# Symbolic Setup
# =============================================================================

# Real but NOT positive: a positive x lets SymPy simplify <x - 0>^-1 to zero, which silently
# drops any support or load sitting at x = 0.
x = Symbol('x', real=True)  # Position along beam
E, I = sp.symbols('E I', positive=True, real=True)  # Elastic modulus, moment of inertia


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PointLoad:
    """Concentrated force. Negative = downward (standard convention)."""
    position: float
    magnitude: float  # Force in your consistent units (e.g., kN)


@dataclass
class PointMoment:
    """Concentrated moment. Positive = counter-clockwise."""
    position: float
    magnitude: float  # Moment in your consistent units (e.g., kN·m)


@dataclass
class DistributedLoad:
    """
    Linearly varying distributed load.

    For UDL: use same value for magnitude_start and magnitude_end.
    For triangular: use different values.

    Negative magnitudes = downward (standard convention).
    """
    start: float
    end: float
    magnitude_start: float  # Intensity at start (kN/m)
    magnitude_end: float    # Intensity at end (kN/m)


# Convenience alias - UDL is just a distributed load with equal ends
def UDL(start: float, end: float, magnitude: float) -> DistributedLoad:
    """Convenience constructor for uniform distributed load."""
    return DistributedLoad(start, end, magnitude, magnitude)


@dataclass
class Support:
    """
    Beam support.

    Types:
        'pin'    - Resists vertical displacement (reaction Ry)
        'roller' - Resists vertical displacement (reaction Ry)
        'fixed'  - Resists displacement AND rotation (reaction Ry + moment M)
    """
    position: float
    type: Literal['pin', 'roller', 'fixed']


@dataclass
class Beam:
    """
    Complete beam definition.

    Attributes:
        length: Total beam length
        supports: List of Support objects
        loads: List of load objects (PointLoad, PointMoment, DistributedLoad, etc.)
        E: Elastic modulus (optional, for numeric deflection)
        I: Moment of inertia (optional, for numeric deflection)
    """
    length: float
    supports: list
    loads: list
    E_val: float | None = None  # Optional numeric value
    I_val: float | None = None  # Optional numeric value


# =============================================================================
# Macaulay Bracket Helpers
# =============================================================================

def macaulay(x_sym, a, n):
    """
    Macaulay bracket: <x - a>^n

    Returns 0 if x < a, else (x - a)^n
    Uses SymPy's SingularityFunction for proper symbolic integration.
    """
    return SingularityFunction(x_sym, a, n)


def format_macaulay_expr(expr):
    """
    Convert SymPy expression to readable Macaulay notation string.

    Replaces SingularityFunction(x, a, n) with <x - a>^n
    """
    s = str(expr)
    # SingularityFunction prints as SingularityFunction(x, a, n)
    # We want <x - a>^n format
    import re

    def replace_sf(match):
        # Extract a and n from SingularityFunction(x, a, n)
        content = match.group(1)
        parts = content.split(',')
        if len(parts) >= 3:
            a = parts[1].strip()
            n = parts[2].strip()
            return f"<x - {a}>^{n}"
        return match.group(0)

    s = re.sub(r'SingularityFunction\(([^)]+)\)', replace_sf, s)
    return s


# =============================================================================
# Beam Solver
# =============================================================================

class BeamSolver:
    """
    Solves beam problems using Macaulay's method.

    Solution process:
        1. Create reaction symbols for each support
        2. Build load function q(x) using Macaulay brackets
        3. Integrate: V(x) = -∫q(x)dx, M(x) = ∫V(x)dx
        4. Set up equilibrium equations (ΣFy=0, ΣM=0)
        5. Set up boundary conditions (y=0 at supports, etc.)
        6. Solve system for reactions and integration constants
        7. Compute slope θ(x) and deflection y(x)
    """

    def __init__(self, beam: Beam):
        self.beam = beam
        self.L = beam.length

        # Reaction symbols (to be solved)
        self.reactions = {}
        self._create_reaction_symbols()

        # Results (populated by solve())
        self.q_expr = None   # Load function
        self.V_expr = None   # Shear
        self.M_expr = None   # Moment
        self.theta_expr = None  # Slope (times EI)
        self.y_expr = None   # Deflection (times EI)

        # Solved numeric expressions (with reactions substituted)
        self.V_solved = None
        self.M_solved = None
        self.theta_solved = None
        self.y_solved = None

        self.solution = {}  # Solved values of reactions and constants

    def _create_reaction_symbols(self):
        """Create symbolic reaction forces/moments for each support."""
        for i, support in enumerate(self.beam.supports):
            pos = support.position
            # Reaction force (always present)
            R_sym = Symbol(f'R_{i}', real=True)
            self.reactions[f'R_{i}'] = {'symbol': R_sym, 'position': pos, 'type': 'force'}

            # Fixed supports also have a moment reaction
            if support.type == 'fixed':
                M_sym = Symbol(f'M_{i}', real=True)
                self.reactions[f'M_{i}'] = {'symbol': M_sym, 'position': pos, 'type': 'moment'}

    def _build_load_function(self):
        """
        Build the distributed load function q(x) using Macaulay brackets.

        Point loads and reactions are represented as Dirac deltas (n=-1).
        Point moments are represented as doublets (n=-2).
        Distributed loads use step functions (n=0) and ramps (n=1).
        """
        q = sp.Integer(0)

        # Add reaction forces (upward positive)
        for key, rxn in self.reactions.items():
            if rxn['type'] == 'force':
                # Point force = intensity * <x-a>^(-1)
                q += rxn['symbol'] * macaulay(x, rxn['position'], -1)
            elif rxn['type'] == 'moment':
                # CCW-positive couple. With sagging-positive M(x), a CCW couple makes the
                # moment drop across it, hence the minus sign (matches the ΣM equation below).
                q -= rxn['symbol'] * macaulay(x, rxn['position'], -2)

        # Add applied loads
        for load in self.beam.loads:
            if isinstance(load, PointLoad):
                # Point load (downward negative in input, so add directly)
                q += load.magnitude * macaulay(x, load.position, -1)

            elif isinstance(load, PointMoment):
                # CCW-positive couple, same sign convention as the reaction moments
                q -= load.magnitude * macaulay(x, load.position, -2)

            elif isinstance(load, DistributedLoad):
                # General distributed load (handles both UDL and triangular)
                # Linear variation from w1 at start to w2 at end
                a, b = load.start, load.end
                w1, w2 = load.magnitude_start, load.magnitude_end
                length = b - a

                if length > 0:
                    # Constant part (step function)
                    q += w1 * macaulay(x, a, 0)
                    q -= w1 * macaulay(x, b, 0)

                    # Ramp part (if not UDL)
                    if w1 != w2:
                        slope = (w2 - w1) / length
                        q += slope * macaulay(x, a, 1)
                        q -= slope * macaulay(x, b, 1)
                        q -= (w2 - w1) * macaulay(x, b, 0)

        self.q_expr = q
        return q

    def _integrate_for_shear_moment(self):
        """
        Integrate load function to get shear and moment.

        V(x) = ∫q(x)dx  (shear is integral of load)
        M(x) = ∫V(x)dx  (moment is integral of shear)

        Note: Sign conventions follow beam theory where:
        - Positive shear: left side up, right side down
        - Positive moment: compression on top (sagging)
        """
        # Shear: integrate q(x)
        # For point loads, integral of <x-a>^(-1) = <x-a>^0
        self.V_expr = integrate(self.q_expr, x)

        # Moment: integrate V(x)
        self.M_expr = integrate(self.V_expr, x)

    def _setup_equations(self):
        """
        Set up equilibrium and boundary condition equations.

        Equilibrium:
            ΣFy = 0 (sum of all vertical forces)
            ΣM = 0  (sum of moments about any point, typically x=0)

        Boundary conditions:
            - Pin/roller: y = 0 at support
            - Fixed: y = 0 AND θ = 0 at support
        """
        equations = []

        # --- Equilibrium: ΣFy = 0 ---
        sum_Fy = sp.Integer(0)

        # Reaction forces
        for key, rxn in self.reactions.items():
            if rxn['type'] == 'force':
                sum_Fy += rxn['symbol']

        # Applied loads
        for load in self.beam.loads:
            if isinstance(load, PointLoad):
                sum_Fy += load.magnitude
            elif isinstance(load, DistributedLoad):
                # Total force = average intensity * length
                avg = (load.magnitude_start + load.magnitude_end) / 2
                sum_Fy += avg * (load.end - load.start)

        equations.append(Eq(sum_Fy, 0))

        # --- Equilibrium: ΣM = 0 (about x=0) ---
        sum_M = sp.Integer(0)

        # Reaction forces and moments
        for key, rxn in self.reactions.items():
            if rxn['type'] == 'force':
                # Moment = force * distance (CCW positive)
                sum_M += rxn['symbol'] * rxn['position']
            elif rxn['type'] == 'moment':
                sum_M += rxn['symbol']

        # Applied loads
        for load in self.beam.loads:
            if isinstance(load, PointLoad):
                sum_M += load.magnitude * load.position
            elif isinstance(load, PointMoment):
                sum_M += load.magnitude
            elif isinstance(load, DistributedLoad):
                # Resultant acts at centroid of trapezoidal distribution
                a, b = load.start, load.end
                w1, w2 = load.magnitude_start, load.magnitude_end
                L = b - a
                if L > 0:
                    total = (w1 + w2) / 2 * L
                    # Centroid of trapezoid from start
                    if w1 + w2 != 0:
                        centroid_from_a = L * (w1 + 2*w2) / (3 * (w1 + w2))
                    else:
                        centroid_from_a = L / 2
                    sum_M += total * (a + centroid_from_a)

        equations.append(Eq(sum_M, 0))

        return equations

    def _setup_boundary_conditions(self):
        """
        Set up boundary condition equations for slope and deflection.

        Uses the integrated expressions:
            EI * θ(x) = ∫M(x)dx + C1
            EI * y(x) = ∫∫M(x)dx + C1*x + C2
        """
        # Integration constants
        C1, C2 = sp.symbols('C1 C2', real=True)

        # Integrate moment for slope and deflection
        # θ = (1/EI) * ∫M dx
        # y = (1/EI) * ∫∫M dx

        theta_EI = integrate(self.M_expr, x) + C1
        y_EI = integrate(theta_EI, x) + C2

        self.theta_expr = theta_EI
        self.y_expr = y_EI

        # Boundary conditions
        bc_equations = []

        for support in self.beam.supports:
            pos = support.position

            # All supports: y = 0
            bc_equations.append(Eq(y_EI.subs(x, pos), 0))

            # Fixed supports: θ = 0 as well
            if support.type == 'fixed':
                bc_equations.append(Eq(theta_EI.subs(x, pos), 0))

        return bc_equations, C1, C2

    def solve(self):
        """
        Solve the beam problem.

        Returns dict with solved reaction values and expressions.
        """
        print("Building load function...")
        self._build_load_function()

        print("Integrating for shear and moment...")
        self._integrate_for_shear_moment()

        print("Setting up equilibrium equations...")
        eq_equations = self._setup_equations()

        print("Setting up boundary conditions...")
        bc_equations, C1, C2 = self._setup_boundary_conditions()

        # Combine all equations
        all_equations = eq_equations + bc_equations

        # Collect all unknowns
        unknowns = [rxn['symbol'] for rxn in self.reactions.values()]
        unknowns.extend([C1, C2])

        print(f"Solving {len(all_equations)} equations for {len(unknowns)} unknowns...")

        # Solve the system
        solution = solve(all_equations, unknowns, dict=True)

        if not solution:
            raise ValueError("Could not solve beam - check if statically determinate")

        self.solution = solution[0]

        # Substitute solution into expressions
        self.V_solved = simplify(self.V_expr.subs(self.solution))
        self.M_solved = simplify(self.M_expr.subs(self.solution))
        self.theta_solved = simplify(self.theta_expr.subs(self.solution))
        self.y_solved = simplify(self.y_expr.subs(self.solution))

        print("Solution complete!\n")

        return self.solution

    def print_results(self):
        """Print solved reactions and expressions in Macaulay notation."""
        print("=" * 60)
        print("BEAM ANALYSIS RESULTS")
        print("=" * 60)

        # Reactions
        print("\n📐 REACTIONS:")
        print("-" * 40)
        for key, rxn in self.reactions.items():
            val = self.solution.get(rxn['symbol'], 'unsolved')
            unit = "force units" if rxn['type'] == 'force' else "moment units"
            pos = rxn['position']
            print(f"  {rxn['symbol']} at x={pos}: {val} ({unit})")

        # Macaulay expressions
        print("\n📝 MACAULAY NOTATION:")
        print("-" * 40)
        print(f"\n  V(x) = {format_macaulay_expr(self.V_solved)}")
        print(f"\n  M(x) = {format_macaulay_expr(self.M_solved)}")
        print(f"\n  EI·θ(x) = {format_macaulay_expr(self.theta_solved)}")
        print(f"\n  EI·y(x) = {format_macaulay_expr(self.y_solved)}")

        print("\n" + "=" * 60)

    def plot(self, num_points: int = 500, show: bool = True, save_path: str | None = None):
        """
        Plot shear, moment, slope, and deflection diagrams.

        Args:
            num_points: Number of points for plotting
            show: Display the plot
            save_path: Save to file if provided
        """
        # Create numeric functions from symbolic expressions
        x_vals = np.linspace(0, self.L, num_points)

        # Convert to numeric functions
        V_func = sp.lambdify(x, self.V_solved, modules=['numpy'])
        M_func = sp.lambdify(x, self.M_solved, modules=['numpy'])
        theta_func = sp.lambdify(x, self.theta_solved, modules=['numpy'])
        y_func = sp.lambdify(x, self.y_solved, modules=['numpy'])

        # Evaluate everywhere, including x = 0 (constant expressions come back as scalars)
        V_vals = np.array([float(V_func(xi)) for xi in x_vals])
        M_vals = np.array([float(M_func(xi)) for xi in x_vals])
        theta_vals = np.array([float(theta_func(xi)) for xi in x_vals])
        y_vals = np.array([float(y_func(xi)) for xi in x_vals])

        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

        # Style settings
        line_color = 'steelblue'
        fill_alpha = 0.3

        # Shear diagram
        axes[0].plot(x_vals, V_vals, color=line_color, linewidth=1.5)
        axes[0].fill_between(x_vals, V_vals, alpha=fill_alpha, color=line_color)
        axes[0].axhline(y=0, color='black', linewidth=0.5)
        axes[0].set_ylabel('Shear V(x)')
        axes[0].set_title('Shear Force Diagram')
        axes[0].grid(True, alpha=0.3)

        # Moment diagram
        axes[1].plot(x_vals, M_vals, color='darkorange', linewidth=1.5)
        axes[1].fill_between(x_vals, M_vals, alpha=fill_alpha, color='darkorange')
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        axes[1].set_ylabel('Moment M(x)')
        axes[1].set_title('Bending Moment Diagram')
        axes[1].grid(True, alpha=0.3)

        # Slope diagram
        axes[2].plot(x_vals, theta_vals, color='green', linewidth=1.5)
        axes[2].axhline(y=0, color='black', linewidth=0.5)
        axes[2].set_ylabel('EI·θ(x)')
        axes[2].set_title('Slope (×EI)')
        axes[2].grid(True, alpha=0.3)

        # Deflection diagram
        axes[3].plot(x_vals, y_vals, color='purple', linewidth=1.5)
        axes[3].axhline(y=0, color='black', linewidth=0.5)
        axes[3].set_ylabel('EI·y(x)')
        axes[3].set_xlabel('Position x')
        axes[3].set_title('Deflection (×EI)')
        axes[3].grid(True, alpha=0.3)

        # Mark supports
        for support in self.beam.supports:
            for ax in axes:
                marker = '^' if support.type in ['pin', 'roller'] else 's'
                ax.axvline(x=support.position, color='red', linestyle='--', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        if show:
            plt.show()

        return fig, axes


# =============================================================================
# Example and Usage
# =============================================================================

def print_usage():
    """Print example syntax and usage information."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     BEAM ANALYSIS TOOL - Macaulay's Method                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
    Define your beam using Python data structures, then call the solver.

EXAMPLE - Simply Supported Beam with Point Load:
────────────────────────────────────────────────
    from beam_analysis import Beam, Support, PointLoad, PointMoment, DistributedLoad, UDL, BeamSolver

    beam = Beam(
        length=10,
        supports=[
            Support(position=0, type='pin'),
            Support(position=10, type='roller'),
        ],
        loads=[
            PointLoad(position=5, magnitude=-10),  # 10 kN downward at midspan
        ]
    )

    solver = BeamSolver(beam)
    solver.solve()
    solver.print_results()
    solver.plot()

EXAMPLE - Cantilever with UDL:
──────────────────────────────
    beam = Beam(
        length=6,
        supports=[
            Support(position=0, type='fixed'),
        ],
        loads=[
            UDL(start=0, end=6, magnitude=-5),  # 5 kN/m downward
        ]
    )

EXAMPLE - Triangular Load:
──────────────────────────
    beam = Beam(
        length=8,
        supports=[
            Support(position=0, type='pin'),
            Support(position=8, type='roller'),
        ],
        loads=[
            # Triangular: 0 kN/m at x=0, increasing to -10 kN/m at x=8
            DistributedLoad(start=0, end=8, magnitude_start=0, magnitude_end=-10),
        ]
    )

EXAMPLE - Overhanging Beam with Multiple Loads:
───────────────────────────────────────────────
    beam = Beam(
        length=12,
        supports=[
            Support(position=2, type='pin'),
            Support(position=10, type='roller'),
        ],
        loads=[
            PointLoad(position=0, magnitude=-8),   # 8 kN at left end
            UDL(start=4, end=8, magnitude=-3),     # 3 kN/m UDL
            PointMoment(position=12, magnitude=10),# 10 kN·m CCW at right end
        ]
    )

LOAD TYPES:
───────────
    PointLoad(position, magnitude)
        Concentrated force. Negative = downward.

    PointMoment(position, magnitude)
        Concentrated moment. Positive = counter-clockwise.

    DistributedLoad(start, end, magnitude_start, magnitude_end)
        Linearly varying distributed load. Negative = downward.
        For UDL, use equal start and end magnitudes.

    UDL(start, end, magnitude)
        Convenience function for uniform distributed load.
        Equivalent to DistributedLoad(start, end, magnitude, magnitude).

SUPPORT TYPES:
──────────────
    'pin'    - Vertical reaction only (allows rotation)
    'roller' - Vertical reaction only (allows rotation + horizontal movement)
    'fixed'  - Vertical reaction + moment reaction (no rotation)

SIGN CONVENTIONS:
─────────────────
    • Loads: Negative = downward, Positive = upward
    • Moments: Positive = counter-clockwise
    • Results shown with standard beam theory conventions

OUTPUT:
───────
    • Solved reaction forces and moments
    • V(x), M(x), θ(x), y(x) in Macaulay notation
    • Shear, moment, slope, and deflection diagrams
""")


def run_example():
    """Run a demonstration example."""
    print("\n" + "─" * 60)
    print("RUNNING EXAMPLE: Simply supported beam with point load")
    print("─" * 60)
    print("Beam: 10m long, pin at x=0, roller at x=10")
    print("Load: 10 kN downward at x=5 (midspan)")
    print("─" * 60 + "\n")

    beam = Beam(
        length=10,
        supports=[
            Support(position=0, type='pin'),
            Support(position=10, type='roller'),
        ],
        loads=[
            PointLoad(position=5, magnitude=-10),
        ]
    )

    solver = BeamSolver(beam)
    solver.solve()
    solver.print_results()
    solver.plot()


if __name__ == '__main__':
    print_usage()

    # Ask if user wants to run example
    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        run_example()
    else:
        print("\nRun with --example to see a demonstration.")
        print("Or import this module and define your own beam.\n")
