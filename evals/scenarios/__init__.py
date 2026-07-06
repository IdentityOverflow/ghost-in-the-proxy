from ..harness import Scenario
from .s1_coherence import SCENARIO as S1
from .s2_tool_heavy import SCENARIO as S2
from .s3_callback import SCENARIO as S3
from .s4_contradiction import SCENARIO as S4
from .s5_salience import SCENARIO as S5
from .s6_containment import SCENARIO as S6
from .s7_fork import SCENARIO as S7
from .s8_interrupt import SCENARIO as S8

ALL_SCENARIOS: list[Scenario] = [S1, S2, S3, S4, S5, S6, S7, S8]


def get_scenarios(ids: list[str] | None = None) -> list[Scenario]:
    if not ids:
        return list(ALL_SCENARIOS)
    by_id = {scenario.id: scenario for scenario in ALL_SCENARIOS}
    missing = [wanted for wanted in ids if wanted not in by_id]
    if missing:
        known = ", ".join(by_id)
        raise KeyError(f"unknown scenario id(s) {missing}; known: {known}")
    return [by_id[wanted] for wanted in ids]
