"""Sovereign economics curriculum: GVC/GRN-USD, land sovereignty, community banking."""
from __future__ import annotations


class SovereignEconomicsCurriculum:
    """Economic literacy rooted in sovereignty, land, and community self-determination."""

    TOKEN_SYSTEM = {
        "EBTK": {
            "name": "East Brooklyn Token",
            "tier": 1,
            "use_case": "In-game rewards, learning achievements, community micro-economy",
            "bridge_rate": "100 EBTK = 1 GVC",
            "backing": "Community engagement (non-speculative)",
        },
        "GVC": {
            "name": "Greenville Coin",
            "tier": 2,
            "use_case": "ICP-native utility token, cross-platform payments",
            "pegged_to": "1 GVC = 1 USD",
            "backing": "Community services and ICP canister operations",
            "bridge_rate": "100 EBTK = 1 GVC",
        },
        "GRN-USD": {
            "name": "Greenville Land-Backed Stablecoin",
            "tier": 2,
            "use_case": "Store of value, land transactions, community bank reserve",
            "pegged_to": "1:1 USD",
            "backing": "53-acre Greenville land collateral + bank deposits",
            "note": "First land-backed stablecoin issued by a sovereign community",
        },
        "WBST": {
            "name": "WealthBridge Service Token",
            "tier": 3,
            "use_case": "Rust/WASM service payments, API access, infrastructure",
            "backing": "Compute services consumed",
        },
    }

    LESSONS = [
        {
            "id": "econ_001",
            "title": "What Is Money? From Cowrie Shells to Crypto",
            "age_group": "k4",
            "duration_min": 25,
            "concepts": ["barter", "currency", "value", "trust"],
            "activity": "Trade virtual goods in East Flatbush market using EBTK tokens.",
            "world_tie": "West African traders used cowrie shells. The Caribbean brought new currencies to Brooklyn.",
            "ebtk_reward": 15,
        },
        {
            "id": "econ_002",
            "title": "Supply, Demand, and the Silk Road",
            "age_group": "g5_8",
            "duration_min": 40,
            "concepts": ["supply", "demand", "price equilibrium", "trade routes"],
            "activity": "Simulate spice prices as supply changes along the Silk Road. Use RAPIDS to chart results.",
            "world_tie": "Silk was worth more than gold in ancient Rome. Why?",
            "ebtk_reward": 30,
            "nvidia_tool": "RAPIDS cuDF + NIM LLM",
        },
        {
            "id": "econ_003",
            "title": "Black Land Loss 1910-1970",
            "age_group": "g9_12",
            "duration_min": 60,
            "concepts": ["land ownership", "redlining", "generational wealth", "sovereignty"],
            "activity": "Analyze USDA data on Black-owned farmland loss. Model generational wealth impact.",
            "reflection": "African Americans owned 16 million acres in 1910. By 1970: 6 million. Analyze causes.",
            "ebtk_reward": 60,
            "nvidia_tool": "RAPIDS cuDF + NIM LLM",
            "resources": ["USDA 2017 Census of Agriculture", "Darity & Mullen (2020) From Here to Equality"],
        },
        {
            "id": "econ_004",
            "title": "Community Banking and the GRN-USD",
            "age_group": "adult",
            "duration_min": 90,
            "concepts": ["stablecoins", "collateralization", "fractional reserve", "sovereign banking"],
            "activity": "Design the reserve requirements and governance rules for a GRN-USD community bank.",
            "reflection": "How does the 53-acre land backing create trust without relying on the US dollar system?",
            "ebtk_reward": 100,
            "nvidia_tool": "NIM LLM + ICP Canister",
            "key_facts": [
                "GRN-USD pegged 1:1 to USD",
                "Backed by 53-acre Greenville, NC land + bank deposits",
                "Issued on ICP blockchain — cannot be seized or deplatformed",
                "Governance by community multi-sig",
            ],
        },
        {
            "id": "econ_005",
            "title": "The EBTK Token Economy",
            "age_group": "g5_8",
            "duration_min": 35,
            "concepts": ["tokens", "reward systems", "learning economy", "community currency"],
            "activity": "Earn EBTK by completing quests. Learn how 100 EBTK bridges to 1 GVC.",
            "world_tie": "East Flatbush residents used informal credit networks (susus/pardners) before modern banking.",
            "ebtk_reward": 25,
            "nvidia_tool": "NIM LLM",
            "token_facts": self.TOKEN_SYSTEM,
        },
    ]

    def get_lesson(self, lesson_id: str) -> dict | None:
        return next((l for l in self.LESSONS if l["id"] == lesson_id), None)

    def token_explainer(self, token: str) -> dict:
        return self.TOKEN_SYSTEM.get(token.upper(), {"error": f"Unknown token: {token}"})

    def full_curriculum(self) -> list[dict]:
        return self.LESSONS
