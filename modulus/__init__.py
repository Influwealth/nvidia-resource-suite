"""NVIDIA Modulus Physics-ML and digital twin framework."""
from .digital_twin import ModulusDigitalTwin, TwinConfig, TwinState
from .pde_solver import PDESolver, NavierStokesSolver, HeatSolver, WaveSolver

try:
    from .physics_ml import PhysicsML, PINNTrainer
except ImportError:
    PhysicsML = None
    PINNTrainer = None

__all__ = [
    "ModulusDigitalTwin",
    "TwinConfig",
    "TwinState",
    "PDESolver",
    "NavierStokesSolver",
    "HeatSolver",
    "WaveSolver",
    "PhysicsML",
    "PINNTrainer",
]
