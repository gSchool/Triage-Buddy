"""Core triage domain: models, safety rules, and the triage service.

Everything in this package must stay free of framework, transport, and
provider concerns. It depends only on the ports (interfaces), never on
concrete adapters.
"""

from triage_buddy.domain.models import (
    EscalationLevel,
    SymptomReport,
    TriageAssessment,
)
from triage_buddy.domain.triage import TriageService

__all__ = [
    "EscalationLevel",
    "SymptomReport",
    "TriageAssessment",
    "TriageService",
]
