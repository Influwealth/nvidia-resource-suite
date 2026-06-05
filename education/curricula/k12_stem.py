"""K-12 STEM curriculum: physics, math, CS, ecology using NVIDIA tools."""
from __future__ import annotations

from typing import Any

from loguru import logger


class K12STEMCurriculum:
    """Full K-12 STEM curriculum with NVIDIA technology integration.

    Each lesson ties GPU compute to tangible learning outcomes.
    Warp for physics, RAPIDS for data science, NIM for tutoring,
    Earth-2 for climate, Modulus for digital twins.
    """

    GRADE_BANDS = {
        "k4": "Kindergarten - Grade 4",
        "g5_8": "Grades 5-8",
        "g9_12": "Grades 9-12",
    }

    LESSONS = {
        "k4": [
            {
                "id": "k4_physics_001",
                "title": "Why Do Things Fall?",
                "subject": "physics",
                "concepts": ["gravity", "falling objects", "trajectory"],
                "nvidia_tool": "NVIDIA Warp (CPU mode)",
                "activity": "Simulate dropping a ball from different heights using the Warp projectile simulator.",
                "world_tie": "In East Flatbush, kids play basketball. Why does the ball arc through the air?",
                "ebtk_reward": 10,
            },
            {
                "id": "k4_math_001",
                "title": "Counting the Silk Road",
                "subject": "mathematics",
                "concepts": ["counting", "addition", "trade"],
                "nvidia_tool": "NIM LLM (tutor mode)",
                "activity": "Ask your AI tutor math questions about trading spices on the Silk Road.",
                "world_tie": "Merchants carried gold, silk, and spices across 7,000 miles.",
                "ebtk_reward": 8,
            },
            {
                "id": "k4_ecology_001",
                "title": "The 53 Acres: A Living World",
                "subject": "ecology",
                "concepts": ["ecosystems", "plants", "soil", "water cycle"],
                "nvidia_tool": "Earth-2 Weather (simplified)",
                "activity": "Explore the weather patterns at the Greenville site and learn how rain grows food.",
                "world_tie": "The sovereign land in Greenville, NC grows food for the community.",
                "ebtk_reward": 10,
            },
        ],
        "g5_8": [
            {
                "id": "g5_8_physics_001",
                "title": "Newton's Laws in the Basketball Court",
                "subject": "physics",
                "concepts": ["Newton's laws", "momentum", "friction", "projectile motion"],
                "nvidia_tool": "NVIDIA Warp particle simulation",
                "activity": "Run a particle simulation of a basketball game. Adjust mass, speed, and angle.",
                "world_tie": "The corner of Flatbush and E 94th has been a basketball hub since the 80s.",
                "ebtk_reward": 20,
                "assessment": "Explain why a heavier ball needs more force to reach the same speed.",
            },
            {
                "id": "g5_8_data_001",
                "title": "Data Science with Block Economics",
                "subject": "mathematics",
                "concepts": ["statistics", "mean", "median", "data visualization"],
                "nvidia_tool": "RAPIDS cuDF (pandas fallback)",
                "activity": "Analyze a dataset of neighborhood business revenues. Find mean, median, top earners.",
                "world_tie": "Economic data from the West Indian Market on Flatbush Avenue.",
                "ebtk_reward": 25,
            },
            {
                "id": "g5_8_cs_001",
                "title": "What is an Algorithm?",
                "subject": "computer_science",
                "concepts": ["algorithms", "loops", "conditionals", "functions"],
                "nvidia_tool": "NIM Code Assist (CodeLlama)",
                "activity": "Write a Python function that calculates trade profits on the Silk Road. Get AI code help.",
                "world_tie": "Al-Khwarizmi, the father of algebra, traveled the Silk Road. His name gave us 'algorithm'.",
                "ebtk_reward": 30,
            },
        ],
        "g9_12": [
            {
                "id": "g9_12_physics_001",
                "title": "Fluid Dynamics: Airflow Over Harlem",
                "subject": "physics",
                "concepts": ["Navier-Stokes", "Reynolds number", "laminar vs turbulent flow"],
                "nvidia_tool": "Modulus Navier-Stokes solver",
                "activity": "Solve the 2D Navier-Stokes equations for airflow around a building. Change Reynolds number.",
                "world_tie": "Jazz music spread from Harlem partly because of the unique acoustic properties of its architecture.",
                "ebtk_reward": 50,
                "assessment": "What happens to airflow when Re > 1000? Describe and simulate.",
            },
            {
                "id": "g9_12_ml_001",
                "title": "Training Your First Neural Network",
                "subject": "computer_science",
                "concepts": ["neural networks", "gradient descent", "loss function", "overfitting"],
                "nvidia_tool": "cuML + NIM LLM",
                "activity": "Train a linear classifier on community economic data. Tune learning rate and watch loss drop.",
                "world_tie": "Predict which Greenville crops will generate the highest yield based on soil and weather data.",
                "ebtk_reward": 60,
            },
            {
                "id": "g9_12_climate_001",
                "title": "Climate Futures: Greenville 2100",
                "subject": "ecology",
                "concepts": ["climate change", "SSP scenarios", "downscaling", "adaptation"],
                "nvidia_tool": "Earth-2 FourCastNet + CorrDiff",
                "activity": "Compare SSP1-2.6 vs SSP5-8.5 for the Greenville site. Generate high-res precipitation maps.",
                "world_tie": "The 53-acre site's food security depends on climate planning.",
                "ebtk_reward": 75,
                "assessment": "Design a climate adaptation plan for the Greenville sovereign site.",
            },
        ],
    }

    def get_lessons(self, grade_band: str) -> list[dict]:
        return self.LESSONS.get(grade_band, [])

    def get_lesson(self, lesson_id: str) -> dict | None:
        for band_lessons in self.LESSONS.values():
            for lesson in band_lessons:
                if lesson["id"] == lesson_id:
                    return lesson
        return None

    def all_lessons(self) -> list[dict]:
        lessons = []
        for band_lessons in self.LESSONS.values():
            lessons.extend(band_lessons)
        return lessons
