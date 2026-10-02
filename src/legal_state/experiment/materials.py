"""Lossless material indexing without model calls, legal rules or reference answers."""

from legal_state.experiment.data import CaseInput
from legal_state.schemas import Fact, LegalState, MaterialSpan


def scoped_material_state(case: CaseInput) -> LegalState:
    """Index the stem and each option as quoted material, not established facts."""
    # Revalidate mutable nested input before computing source positions.
    case = CaseInput.model_validate(case.model_dump())
    facts = [
        Fact(
            id="F1",
            content=case.stem,
            source="question_material",
            material=MaterialSpan(start=0, end=len(case.stem)),
        )
    ]
    cursor = len(case.stem)
    for index, letter in enumerate("ABCD", start=2):
        cursor += len(f"{letter}:")
        content = case.options[letter]
        end = cursor + len(content)
        if case.question[cursor:end] != content:
            raise ValueError("Material does not match original question span")
        facts.append(
            Fact(
                id=f"F{index}",
                content=content,
                source="question_material",
                material=MaterialSpan(start=cursor, end=end, scope=letter),
            )
        )
        cursor = end
    return LegalState(facts=facts)
