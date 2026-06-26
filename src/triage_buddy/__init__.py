"""Triage Buddy: escalation advice for medical symptoms.

Package layout follows a hexagonal (ports & adapters) architecture:

- ``domain``   : the core triage logic. No framework, transport, or provider imports.
- ``ports``    : the interfaces the core depends on (e.g. the LLM provider port).
- ``adapters`` : concrete implementations of ports and entry points (mock LLM, CLI).
"""

__version__ = "0.1.0"
