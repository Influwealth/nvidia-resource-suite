"""Quest engine: EBTK rewards, progress tracking, completion certificates."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class QuestState(str, Enum):
    LOCKED = "locked"
    AVAILABLE = "available"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QuestReward:
    ebtk: float = 0.0
    gvc: float = 0.0
    badge: str | None = None
    unlock_quest_ids: list[str] = field(default_factory=list)
    certificate: bool = False


@dataclass
class QuestProgress:
    quest_id: str
    learner_id: str
    state: QuestState = QuestState.AVAILABLE
    current_step: int = 0
    total_steps: int = 1
    started_at: float | None = None
    completed_at: float | None = None
    score: float = 0.0
    reward_claimed: bool = False
    attempts: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Quest:
    quest_id: str
    world: str
    title: str
    description: str
    difficulty: str      # easy | medium | hard | legendary
    duration_min: int
    steps: list[dict]
    reward: QuestReward
    min_age_group: str = "k4"
    prerequisites: list[str] = field(default_factory=list)
    max_attempts: int = 3
    languages: list[str] = field(default_factory=lambda: ["en"])


class WorldQuestEngine:
    """Quest management engine with EBTK token reward distribution.

    Manages quest state, validates completions, issues rewards,
    and generates completion certificates via NIM.
    """

    def __init__(self, nim_client=None, ebtk_ledger=None):
        self._nim = nim_client
        self._ledger = ebtk_ledger
        self._quests: dict[str, Quest] = {}
        self._progress: dict[str, dict[str, QuestProgress]] = {}  # learner_id → quest_id → progress
        self._load_default_quests()

    def _load_default_quests(self):
        self.register_quest(Quest(
            quest_id="eastflatbush_458_frequency",
            world="east_flatbush",
            title="The 458 Frequency",
            description="Uncover the sonic history of 458 E 94th St and produce your own beat.",
            difficulty="medium",
            duration_min=30,
            steps=[
                {"step": 1, "title": "Oral History Archive", "type": "listen", "asset": "458_e94_oral_history"},
                {"step": 2, "title": "Beat Construction", "type": "create", "tool": "audio2face"},
                {"step": 3, "title": "Community Screening", "type": "share", "platform": "synapz"},
            ],
            reward=QuestReward(ebtk=50.0, badge="458_frequency_graduate", certificate=True),
            languages=["en", "es", "ht"],
        ))
        self.register_quest(Quest(
            quest_id="greenville_community_bank",
            world="greenville",
            title="Build the Community Bank",
            description="Design a sovereign community bank for the 53-acre Greenville site.",
            difficulty="hard",
            duration_min=60,
            steps=[
                {"step": 1, "title": "Land Survey", "type": "simulation", "tool": "modulus_twin"},
                {"step": 2, "title": "GRN-USD Economics", "type": "study", "module": "sovereign_economics"},
                {"step": 3, "title": "Bank Charter Draft", "type": "write", "tool": "nim_llm"},
                {"step": 4, "title": "Community Presentation", "type": "present", "platform": "vr_cockpit"},
            ],
            reward=QuestReward(ebtk=100.0, gvc=1.0, badge="community_banker", certificate=True),
        ))
        self.register_quest(Quest(
            quest_id="silk_road_merchant",
            world="ancient_silk_road",
            title="The Merchant's Journey",
            description="Trade goods from Chang'an to Rome, mastering mathematics and economics along the way.",
            difficulty="medium",
            duration_min=45,
            steps=[
                {"step": 1, "title": "Route Planning", "type": "geography", "tool": "earth2_forecast"},
                {"step": 2, "title": "Currency Exchange", "type": "math", "tool": "rapids_cudf"},
                {"step": 3, "title": "Arrival in Samarkand", "type": "narrative", "tool": "nim_llm"},
            ],
            reward=QuestReward(ebtk=45.0, badge="silk_road_merchant"),
            languages=["en", "ar", "zh", "fa", "tr"],
        ))
        self.register_quest(Quest(
            quest_id="harlem_voices",
            world="harlem_renaissance",
            title="Voices of the Renaissance",
            description="Transcribe and analyze speeches and poems from Harlem Renaissance leaders.",
            difficulty="easy",
            duration_min=25,
            steps=[
                {"step": 1, "title": "Audio Archive", "type": "listen", "tool": "nemo_asr"},
                {"step": 2, "title": "Poem Analysis", "type": "study", "tool": "nim_llm"},
                {"step": 3, "title": "Your Voice", "type": "record", "tool": "nemo_tts"},
            ],
            reward=QuestReward(ebtk=30.0, badge="harlem_voice"),
            languages=["en", "fr"],
        ))

    def register_quest(self, quest: Quest):
        self._quests[quest.quest_id] = quest
        logger.debug(f"Registered quest: {quest.quest_id}")

    def start_quest(self, learner_id: str, quest_id: str) -> QuestProgress:
        quest = self._quests.get(quest_id)
        if not quest:
            raise ValueError(f"Quest {quest_id} not found")
        learner_progress = self._progress.setdefault(learner_id, {})
        existing = learner_progress.get(quest_id)
        if existing and existing.state == QuestState.COMPLETED:
            return existing
        if existing and existing.attempts >= quest.max_attempts:
            raise RuntimeError(f"Max attempts ({quest.max_attempts}) reached for quest {quest_id}")
        progress = QuestProgress(
            quest_id=quest_id,
            learner_id=learner_id,
            state=QuestState.IN_PROGRESS,
            total_steps=len(quest.steps),
            started_at=time.time(),
            attempts=(existing.attempts + 1) if existing else 1,
        )
        learner_progress[quest_id] = progress
        logger.info(f"Learner {learner_id} started quest {quest_id}")
        return progress

    def advance_step(
        self,
        learner_id: str,
        quest_id: str,
        step_score: float = 100.0,
    ) -> QuestProgress:
        progress = self._get_progress(learner_id, quest_id)
        if progress.state != QuestState.IN_PROGRESS:
            raise RuntimeError(f"Quest {quest_id} is not in progress")
        progress.current_step += 1
        progress.score = (progress.score * (progress.current_step - 1) + step_score) / progress.current_step
        if progress.current_step >= progress.total_steps:
            progress.state = QuestState.COMPLETED
            progress.completed_at = time.time()
            logger.info(f"Learner {learner_id} completed quest {quest_id} with score {progress.score:.1f}")
        return progress

    def claim_reward(self, learner_id: str, quest_id: str) -> QuestReward:
        progress = self._get_progress(learner_id, quest_id)
        if progress.state != QuestState.COMPLETED:
            raise RuntimeError("Quest not completed")
        if progress.reward_claimed:
            raise RuntimeError("Reward already claimed")
        quest = self._quests[quest_id]
        reward = quest.reward
        if self._ledger:
            self._ledger.credit(learner_id, reward.ebtk, "EBTK")
            if reward.gvc > 0:
                self._ledger.credit(learner_id, reward.gvc, "GVC")
        progress.reward_claimed = True
        logger.info(f"Reward claimed: {reward.ebtk} EBTK, {reward.gvc} GVC → {learner_id}")
        return reward

    def generate_certificate(
        self,
        learner_id: str,
        quest_id: str,
    ) -> dict:
        progress = self._get_progress(learner_id, quest_id)
        quest = self._quests.get(quest_id)
        if not quest or not quest.reward.certificate:
            return {"certificate": False}
        cert = {
            "learner_id": learner_id,
            "quest": quest.title,
            "world": quest.world,
            "score": round(progress.score, 1),
            "completed_at": progress.completed_at,
            "badge": quest.reward.badge,
            "ebtk_earned": quest.reward.ebtk,
            "certificate_id": str(uuid.uuid4()),
            "issued_by": "NVIDIA Resource Suite — Public Global Educational Platform",
        }
        if self._nim:
            try:
                cert["narrative"] = self._nim.generate(
                    f"Write a 2-sentence congratulatory message for a student who completed the '{quest.title}' quest "
                    f"in the {quest.world} world with a score of {progress.score:.0f}%.",
                    max_new_tokens=100,
                )
            except Exception:
                cert["narrative"] = f"Congratulations on completing {quest.title}!"
        return cert

    def get_available_quests(
        self,
        learner_id: str,
        world: str | None = None,
    ) -> list[Quest]:
        learner_progress = self._progress.get(learner_id, {})
        quests = list(self._quests.values())
        if world:
            quests = [q for q in quests if q.world == world]
        available = []
        for q in quests:
            prog = learner_progress.get(q.quest_id)
            if prog and prog.state == QuestState.COMPLETED:
                continue
            prereqs_met = all(
                learner_progress.get(p, QuestProgress("", "")).state == QuestState.COMPLETED
                for p in q.prerequisites
            )
            if prereqs_met:
                available.append(q)
        return available

    def learner_summary(self, learner_id: str) -> dict:
        learner_progress = self._progress.get(learner_id, {})
        completed = [p for p in learner_progress.values() if p.state == QuestState.COMPLETED]
        total_ebtk = sum(
            self._quests[p.quest_id].reward.ebtk
            for p in completed
            if p.quest_id in self._quests and p.reward_claimed
        )
        return {
            "learner_id": learner_id,
            "quests_completed": len(completed),
            "quests_in_progress": sum(1 for p in learner_progress.values() if p.state == QuestState.IN_PROGRESS),
            "total_ebtk_earned": total_ebtk,
            "badges": [
                self._quests[p.quest_id].reward.badge
                for p in completed
                if p.quest_id in self._quests and self._quests[p.quest_id].reward.badge
            ],
        }

    def _get_progress(self, learner_id: str, quest_id: str) -> QuestProgress:
        progress = self._progress.get(learner_id, {}).get(quest_id)
        if not progress:
            raise ValueError(f"No progress found for learner {learner_id} / quest {quest_id}")
        return progress
