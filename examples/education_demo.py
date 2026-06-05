#!/usr/bin/env python3
"""Education Demo: full learning session with quest, adaptive lesson, quiz, and certificate."""
import os
from loguru import logger

from nim.client import NIMClient
from nim.llm import LLMClient
from nim.embedding import EmbeddingClient
from education.curriculum import CurriculumEngine, LearnerProfile, AgeGroup
from education.world_quest import WorldQuestEngine
from education.curricula import K12STEMCurriculum, AILiteracyCurriculum, SovereignEconomicsCurriculum
from warp.physics import WarpPhysics

API_KEY = os.environ.get("NVIDIA_API_KEY", "")


def demo_learner_profile():
    print("\n=== Create Learner Profile ===")
    profile = LearnerProfile(
        age_group=AgeGroup.G9_12,
        language="en",
        world="east_flatbush",
        skills={"physics": 45, "mathematics": 62, "computer_science": 30, "economics": 55},
    )
    print(f"Learner: {profile.learner_id}")
    print(f"Age group: {profile.age_group}")
    print(f"World: {profile.world}")
    print(f"Skill scores: {profile.skills}")
    return profile


def demo_adaptive_lesson(profile: LearnerProfile):
    print("\n=== Generate Adaptive Lesson ===")
    nim = NIMClient(api_key=API_KEY)
    llm = LLMClient(nim)
    engine = CurriculumEngine(llm)

    # Suggest next lesson based on weakest skill
    suggestion = engine.suggest_next_lesson(profile)
    print(f"Suggested subject: {suggestion['recommended_subject']}")
    print(f"Reason: {suggestion['reason']}")
    print(f"EBTK potential: {suggestion['ebtk_potential']}")

    # Generate the lesson
    lesson = engine.generate_lesson(
        profile=profile,
        subject=suggestion["recommended_subject"],
        topic="algorithms and complexity",
        duration_min=45,
    )
    print(f"\nLesson: '{lesson.title}'")
    print(f"NVIDIA tools: {', '.join(lesson.nvidia_tech_used)}")
    print(f"Objectives:")
    for obj in lesson.objectives:
        print(f"  • {obj}")
    print(f"Activities:")
    for act in lesson.activities:
        print(f"  [{act['type']}] {act['title']} ({act['duration_min']} min) — {act['nvidia_tool']}")
    return lesson


def demo_k12_curriculum():
    print("\n=== K-12 STEM Curriculum ===")
    k12 = K12STEMCurriculum()
    for band in ["k4", "g5_8", "g9_12"]:
        lessons = k12.get_lessons(band)
        print(f"\nGrade band: {k12.GRADE_BANDS[band]} ({len(lessons)} lessons)")
        for lesson in lessons:
            print(f"  [{lesson['subject']}] {lesson['title']} — {lesson['ebtk_reward']} EBTK")
            print(f"    Tool: {lesson['nvidia_tool']}")


def demo_ai_literacy():
    print("\n=== AI Literacy Curriculum ===")
    ai_lit = AILiteracyCurriculum()
    path = ai_lit.learning_path()
    print(f"Learning path ({len(path)} modules): {' → '.join(path)}")
    for module_id in path[:3]:
        m = ai_lit.get_module(module_id)
        print(f"\n  {m['id']}: {m['title']} ({m['age_group']}, {m['duration_min']} min)")
        print(f"  Concepts: {', '.join(m['concepts'][:3])}")
        print(f"  NVIDIA: {m['nvidia_tool']}")


def demo_sovereign_economics():
    print("\n=== Sovereign Economics Curriculum ===")
    econ = SovereignEconomicsCurriculum()
    print("Token system:")
    for token, info in econ.TOKEN_SYSTEM.items():
        print(f"  {token}: {info['name']} — {info['use_case'][:60]}")
    print(f"\nLessons:")
    for lesson in econ.full_curriculum():
        print(f"  [{lesson['age_group']}] {lesson['title']} ({lesson['duration_min']} min, {lesson['ebtk_reward']} EBTK)")


def demo_physics_simulation():
    print("\n=== NVIDIA Warp Physics (Educational Demo) ===")
    warp = WarpPhysics()
    print(f"Warp available: {warp.available}")

    # Projectile motion
    trajectory = warp.simulate_projectile(
        initial_position=(0.0, 0.0, 0.0),
        initial_velocity=(15.0, 25.0, 0.0),  # basketball throw
        gravity=-9.81,
        dt=0.05,
        max_time=5.0,
    )
    peak = max(trajectory, key=lambda p: p["y"])
    landing = trajectory[-1]
    print(f"Basketball throw simulation:")
    print(f"  Initial velocity: 15 m/s horizontal, 25 m/s vertical")
    print(f"  Peak height: {peak['y']:.2f}m at t={peak['t']:.2f}s, x={peak['x']:.2f}m")
    print(f"  Landing: x={landing['x']:.2f}m, t={landing['t']:.2f}s ({len(trajectory)} points)")

    # Elastic collision
    collision = warp.rigid_body_collision(
        body_a_pos=(0, 0, 0), body_b_pos=(2, 0, 0),
        body_a_vel=(5, 0, 0), body_b_vel=(-2, 0, 0),
        mass_a=1.0, mass_b=2.0,
    )
    print(f"\nElastic collision (1kg vs 2kg):")
    print(f"  Body A after: vx={collision['body_a_velocity_after'][0]:.2f} m/s")
    print(f"  Body B after: vx={collision['body_b_velocity_after'][0]:.2f} m/s")
    print(f"  Energy conservation: {collision['energy_before']:.3f} → {collision['energy_after']:.3f} J")


def demo_quest_session():
    print("\n=== Complete Quest Session ===")
    nim = NIMClient(api_key=API_KEY)
    llm = LLMClient(nim)
    engine = WorldQuestEngine(llm)

    learner_id = "demo_learner_001"
    quest_id = "eastflatbush_458_frequency"

    print(f"Starting quest: 'The 458 Frequency'")
    progress = engine.start_quest(learner_id, quest_id)
    print(f"  State: {progress.state} | Steps: {progress.total_steps}")

    for step_num in range(progress.total_steps):
        score = 85.0 + step_num * 5  # simulate improving performance
        progress = engine.advance_step(learner_id, quest_id, step_score=score)
        print(f"  Step {progress.current_step}/{progress.total_steps}: score={progress.score:.1f}")

    print(f"  Final state: {progress.state}")

    if progress.state == "completed":
        reward = engine.claim_reward(learner_id, quest_id)
        print(f"  Reward: {reward.ebtk} EBTK | Badge: {reward.badge}")
        cert = engine.generate_certificate(learner_id, quest_id)
        print(f"  Certificate ID: {cert.get('certificate_id', 'N/A')}")
        print(f"  Issued by: {cert.get('issued_by', '')}")


if __name__ == "__main__":
    profile = demo_learner_profile()
    demo_adaptive_lesson(profile)
    demo_k12_curriculum()
    demo_ai_literacy()
    demo_sovereign_economics()
    demo_physics_simulation()
    demo_quest_session()
    print("\n=== Education demo complete! ===")
