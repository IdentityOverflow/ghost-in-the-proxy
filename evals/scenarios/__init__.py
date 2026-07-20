from ..harness import Scenario
from .s1_coherence import SCENARIO as S1
from .s2_tool_heavy import SCENARIO as S2
from .s3_callback import SCENARIO as S3
from .s4_contradiction import SCENARIO as S4
from .s5_salience import SCENARIO as S5
from .s6_containment import SCENARIO as S6
from .s7_fork import SCENARIO as S7
from .s8_interrupt import SCENARIO as S8
from .s9_verbatim import SCENARIO as S9
from .s10_tool_tax import SCENARIO as S10
from .s11_time import SCENARIO as S11
from .s12_semantic_callback import SCENARIO as S12
from .s13_sequence_recall import SCENARIO as S13

ALL_SCENARIOS: list[Scenario] = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13]


def get_scenarios(ids: list[str] | None = None) -> list[Scenario]:
    if not ids:
        return list(ALL_SCENARIOS)
    by_id = {scenario.id: scenario for scenario in ALL_SCENARIOS}
    missing = [wanted for wanted in ids if wanted not in by_id]
    if missing:
        known = ", ".join(by_id)
        raise KeyError(f"unknown scenario id(s) {missing}; known: {known}")
    return [by_id[wanted] for wanted in ids]
