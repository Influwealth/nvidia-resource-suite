"""NVIDIA Warp GPU kernels for particles, cloth, rigid bodies, educational physics."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

try:
    import warp as wp
    import numpy as np
    WARP_AVAILABLE = True
    wp.init()
    logger.info(f"NVIDIA Warp {wp.__version__} initialized")
except ImportError:
    WARP_AVAILABLE = False
    logger.warning("NVIDIA Warp not installed — physics running in CPU simulation mode")


@dataclass
class WarpKernel:
    """Descriptor for a custom GPU kernel written with @wp.kernel decorator."""
    name: str
    description: str
    fn: Callable | None = None
    device: str = "cuda"

    def launch(self, dim: int | tuple, inputs: list, outputs: list):
        if not WARP_AVAILABLE or self.fn is None:
            logger.info(f"[MOCK] Kernel '{self.name}' launched with dim={dim}")
            return
        wp.launch(self.fn, dim=dim, inputs=inputs, outputs=outputs, device=self.device)


@dataclass
class ParticleState:
    positions: Any  # wp.array or list of (x,y,z)
    velocities: Any
    masses: Any
    num_particles: int


@dataclass
class ClothSimConfig:
    num_particles_x: int = 32
    num_particles_y: int = 32
    particle_mass: float = 0.1
    spring_ke: float = 1000.0  # elastic stiffness
    spring_kd: float = 0.5     # damping
    gravity: tuple = (0.0, -9.81, 0.0)
    dt: float = 1.0 / 60.0


class ParticleSimulation:
    """GPU-accelerated particle simulation using NVIDIA Warp.

    Used for educational physics demos: fluid, gas, crowds, starfields.
    Falls back to pure Python when Warp is unavailable.
    """

    def __init__(
        self,
        num_particles: int = 1000,
        bounds: tuple = ((-5, -5, -5), (5, 5, 5)),
        gravity: tuple = (0.0, -9.81, 0.0),
        device: str = "cuda",
    ):
        self.num_particles = num_particles
        self.bounds = bounds
        self.gravity = gravity
        self.device = device if WARP_AVAILABLE else "cpu"
        self._state = self._init_state()

    def _init_state(self):
        import random
        lo, hi = self.bounds
        positions = [
            [random.uniform(lo[0], hi[0]), random.uniform(lo[1], hi[1]), random.uniform(lo[2], hi[2])]
            for _ in range(self.num_particles)
        ]
        velocities = [[0.0, 0.0, 0.0]] * self.num_particles
        masses = [1.0] * self.num_particles

        if not WARP_AVAILABLE:
            return ParticleState(positions=positions, velocities=velocities, masses=masses, num_particles=self.num_particles)

        return ParticleState(
            positions=wp.array(positions, dtype=wp.vec3, device=self.device),
            velocities=wp.array(velocities, dtype=wp.vec3, device=self.device),
            masses=wp.array(masses, dtype=float, device=self.device),
            num_particles=self.num_particles,
        )

    def step(self, dt: float = 1.0 / 60.0) -> Any:
        """Advance particle simulation by one timestep."""
        if WARP_AVAILABLE:
            return self._warp_step(dt)
        return self._cpu_step(dt)

    def _cpu_step(self, dt: float):
        lo, hi = self.bounds
        gx, gy, gz = self.gravity
        new_positions = []
        new_velocities = []
        for pos, vel in zip(self._state.positions, self._state.velocities):
            vx = vel[0] + gx * dt
            vy = vel[1] + gy * dt
            vz = vel[2] + gz * dt
            x = pos[0] + vx * dt
            y = pos[1] + vy * dt
            z = pos[2] + vz * dt
            # Bounce at bounds
            if x < lo[0] or x > hi[0]: vx *= -0.8
            if y < lo[1] or y > hi[1]: vy *= -0.8; y = max(lo[1], min(hi[1], y))
            if z < lo[2] or z > hi[2]: vz *= -0.8
            new_positions.append([x, y, z])
            new_velocities.append([vx, vy, vz])
        self._state.positions = new_positions
        self._state.velocities = new_velocities
        return self._state

    def _warp_step(self, dt: float):
        if not WARP_AVAILABLE:
            return self._state

        @wp.kernel
        def integrate_particles(
            pos: wp.array(dtype=wp.vec3),
            vel: wp.array(dtype=wp.vec3),
            gravity: wp.vec3,
            dt: float,
        ):
            tid = wp.tid()
            v = vel[tid] + gravity * dt
            p = pos[tid] + v * dt
            vel[tid] = v
            pos[tid] = p

        grav_vec = wp.vec3(self.gravity[0], self.gravity[1], self.gravity[2])
        wp.launch(
            integrate_particles,
            dim=self.num_particles,
            inputs=[self._state.positions, self._state.velocities, grav_vec, dt],
            device=self.device,
        )
        return self._state

    def get_positions_list(self) -> list:
        if WARP_AVAILABLE and hasattr(self._state.positions, "numpy"):
            return self._state.positions.numpy().tolist()
        return self._state.positions


class ClothSimulation:
    """GPU cloth simulation using NVIDIA Warp.

    Educational use: demonstrate textile physics, flags, garments.
    """

    def __init__(self, config: ClothSimConfig | None = None, device: str = "cuda"):
        self.config = config or ClothSimConfig()
        self.device = device if WARP_AVAILABLE else "cpu"
        self._model = None
        self._state = None
        if WARP_AVAILABLE:
            self._build_model()

    def _build_model(self):
        try:
            import warp.sim as wps
            builder = wps.ModelBuilder()
            cfg = self.config
            builder.add_cloth_grid(
                pos=(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi / 2),
                vel=(0.0, 0.0, 0.0),
                dim_x=cfg.num_particles_x,
                dim_y=cfg.num_particles_y,
                cell_x=0.1,
                cell_y=0.1,
                mass=cfg.particle_mass,
                tri_ke=cfg.spring_ke,
                tri_kd=cfg.spring_kd,
            )
            self._model = builder.finalize(self.device)
            self._model.gravity = wp.vec3(*self.config.gravity)
            self._state = self._model.state()
            logger.info(f"Cloth simulation ready: {cfg.num_particles_x}x{cfg.num_particles_y} particles")
        except Exception as e:
            logger.warning(f"Cloth sim build failed: {e}")

    def step(self) -> dict:
        """Advance cloth simulation by one timestep."""
        if not WARP_AVAILABLE or self._model is None:
            return {"mock": True, "note": "Warp not available"}
        try:
            import warp.sim as wps
            integrator = wps.SemiImplicitIntegrator()
            self._state = integrator.simulate(self._model, self._state, self._state, self.config.dt)
            return {"num_particles": self.config.num_particles_x * self.config.num_particles_y, "dt": self.config.dt}
        except Exception as e:
            return {"error": str(e)}

    def get_vertex_positions(self) -> list:
        if not WARP_AVAILABLE or self._state is None:
            rows, cols = self.config.num_particles_x, self.config.num_particles_y
            return [[float(i), float(j), 0.0] for i in range(rows) for j in range(cols)]
        return self._state.particle_q.numpy().tolist()


class WarpPhysics:
    """High-level NVIDIA Warp physics facade for educational applications."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.available = WARP_AVAILABLE

    def simulate_projectile(
        self,
        initial_position: tuple = (0.0, 0.0, 0.0),
        initial_velocity: tuple = (10.0, 20.0, 0.0),
        gravity: float = -9.81,
        dt: float = 0.01,
        max_time: float = 5.0,
    ) -> list[dict]:
        """Projectile motion — runs on CPU or GPU."""
        trajectory = []
        x, y, z = initial_position
        vx, vy, vz = initial_velocity
        t = 0.0
        while t <= max_time and y >= 0:
            trajectory.append({"t": round(t, 3), "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
            vy += gravity * dt
            x += vx * dt; y += vy * dt; z += vz * dt
            t += dt
        return trajectory

    def create_particle_sim(self, num_particles: int = 1000, **kwargs) -> ParticleSimulation:
        return ParticleSimulation(num_particles=num_particles, device=self.device, **kwargs)

    def create_cloth_sim(self, config: ClothSimConfig | None = None) -> ClothSimulation:
        return ClothSimulation(config=config, device=self.device)

    def rigid_body_collision(
        self,
        body_a_pos: tuple,
        body_b_pos: tuple,
        body_a_vel: tuple = (0.0, 0.0, 0.0),
        body_b_vel: tuple = (0.0, 0.0, 0.0),
        mass_a: float = 1.0,
        mass_b: float = 1.0,
        restitution: float = 0.8,
    ) -> dict:
        """Elastic collision — pure Python for cross-platform education."""
        # 1D elastic collision along x-axis
        vax, vbx = body_a_vel[0], body_b_vel[0]
        ma, mb = mass_a, mass_b
        vax_new = ((ma - mb) * vax + 2 * mb * vbx) / (ma + mb)
        vbx_new = ((mb - ma) * vbx + 2 * ma * vax) / (ma + mb)
        return {
            "body_a_velocity_after": (vax_new * restitution, body_a_vel[1], body_a_vel[2]),
            "body_b_velocity_after": (vbx_new * restitution, body_b_vel[1], body_b_vel[2]),
            "energy_before": 0.5 * ma * vax**2 + 0.5 * mb * vbx**2,
            "energy_after": 0.5 * ma * vax_new**2 + 0.5 * mb * vbx_new**2,
        }

    def status(self) -> dict:
        return {
            "warp_available": WARP_AVAILABLE,
            "device": self.device,
            "version": getattr(wp, "__version__", "N/A") if WARP_AVAILABLE else "N/A",
        }
