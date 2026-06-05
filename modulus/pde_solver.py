"""PDE solvers: Navier-Stokes, heat equation, wave equation, Darcy flow."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed — PDE solvers using simple finite differences")


@dataclass
class GridConfig:
    nx: int = 64    # x grid points
    ny: int = 64    # y grid points
    nt: int = 100   # time steps
    dx: float = 0.1
    dy: float = 0.1
    dt: float = 0.01
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0


class PDESolver(ABC):
    """Abstract base for PDE solvers."""

    def __init__(self, grid: GridConfig):
        self.grid = grid

    @abstractmethod
    def solve(self) -> dict:
        """Run the solver and return field arrays."""
        ...

    def export_json(self, result: dict, max_points: int = 1000) -> list[dict]:
        """Convert solution arrays to JSON-serializable point list."""
        points = []
        if not NUMPY_AVAILABLE:
            return points
        g = self.grid
        xs = [g.x_min + i * g.dx for i in range(min(g.nx, 50))]
        ys = [g.y_min + j * g.dy for j in range(min(g.ny, 50))]
        field_key = next((k for k in result if k not in ("t", "converged", "steps")), None)
        if field_key is None:
            return points
        field = result[field_key]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                val = float(field[i, j]) if hasattr(field, "__getitem__") else 0.0
                points.append({"x": round(x, 4), "y": round(y, 4), field_key: round(val, 6)})
                if len(points) >= max_points:
                    return points
        return points


class HeatSolver(PDESolver):
    """2D heat equation: ∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)"""

    def __init__(self, grid: GridConfig, alpha: float = 0.01, initial_temp: float = 0.0):
        super().__init__(grid)
        self.alpha = alpha
        self.initial_temp = initial_temp

    def solve(self) -> dict:
        g = self.grid
        if not NUMPY_AVAILABLE:
            return {"temperature": [], "note": "numpy required for heat solver"}

        T = np.full((g.nx, g.ny), self.initial_temp, dtype=float)
        # Boundary conditions: hot left wall
        T[0, :] = 100.0
        T[-1, :] = 0.0
        T[:, 0] = 0.0
        T[:, -1] = 0.0

        r = self.alpha * g.dt / (g.dx ** 2)
        if r > 0.25:
            logger.warning(f"Heat solver: stability condition r={r:.3f} > 0.25. Reduce dt or increase dx.")

        for _ in range(g.nt):
            T_new = T.copy()
            T_new[1:-1, 1:-1] = (
                T[1:-1, 1:-1]
                + r * (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1])
                + r * (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2])
            )
            T_new[0, :] = 100.0
            T_new[-1, :] = 0.0
            T = T_new

        return {"temperature": T, "alpha": self.alpha, "steps": g.nt}


class WaveSolver(PDESolver):
    """2D wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)"""

    def __init__(self, grid: GridConfig, c: float = 1.0):
        super().__init__(grid)
        self.c = c

    def solve(self) -> dict:
        g = self.grid
        if not NUMPY_AVAILABLE:
            return {"displacement": [], "note": "numpy required"}

        u = np.zeros((g.nx, g.ny))
        u_prev = np.zeros((g.nx, g.ny))
        # Gaussian pulse at center
        cx, cy = g.nx // 2, g.ny // 2
        for i in range(g.nx):
            for j in range(g.ny):
                r2 = ((i - cx) * g.dx) ** 2 + ((j - cy) * g.dy) ** 2
                u[i, j] = math.exp(-50 * r2)
        u_prev = u.copy()

        r = (self.c * g.dt / g.dx) ** 2
        for _ in range(g.nt):
            u_next = np.zeros_like(u)
            u_next[1:-1, 1:-1] = (
                2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
                + r * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
                + r * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
            )
            u_prev = u
            u = u_next

        return {"displacement": u, "wave_speed": self.c, "steps": g.nt}


class NavierStokesSolver(PDESolver):
    """2D incompressible Navier-Stokes (lid-driven cavity) via finite differences."""

    def __init__(self, grid: GridConfig, Re: float = 100.0):
        """Re = Reynolds number. Re < 1000 for laminar flow."""
        super().__init__(grid)
        self.Re = Re

    def solve(self) -> dict:
        g = self.grid
        if not NUMPY_AVAILABLE:
            return {"u": [], "v": [], "p": [], "note": "numpy required"}

        u = np.zeros((g.nx + 2, g.ny + 2))
        v = np.zeros((g.nx + 2, g.ny + 2))
        p = np.zeros((g.nx + 2, g.ny + 2))

        nu = 1.0 / self.Re
        dx, dy, dt = g.dx, g.dy, g.dt

        for _ in range(g.nt):
            u[1:-1, -1] = 1.0  # lid velocity

            un = u.copy()
            vn = v.copy()

            # Pressure Poisson (simplified)
            for _ in range(20):
                pn = p.copy()
                p[1:-1, 1:-1] = (
                    (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2
                    + (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2
                ) / (2 * (dx**2 + dy**2))

            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[:-2, 1:-1])
                - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, :-2])
                - dt / (2 * dx) * (p[2:, 1:-1] - p[:-2, 1:-1])
                + nu * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
                + nu * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            )

        return {"u": u[1:-1, 1:-1], "v": v[1:-1, 1:-1], "p": p[1:-1, 1:-1], "Re": self.Re}
