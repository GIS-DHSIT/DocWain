from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Fact:
    label: str
    value: str
    basis: str
    doc_name: str
    fact_type: str


@dataclass
class FactCache:
    entities: Dict[str, List[str]] = field(default_factory=lambda: {"person": [], "org": [], "date": [], "money": [], "id": []})
    sections: Dict[str, List[str]] = field(default_factory=dict)
    key_values: Dict[str, List[str]] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    doc_names: List[str] = field(default_factory=list)

    def add_entity(self, kind: str, value: str, basis: str, doc_name: str) -> None:
        if value and value not in self.entities.setdefault(kind, []):
            self.entities[kind].append(value)
            self.facts.append(Fact(label=kind, value=value, basis=basis, doc_name=doc_name, fact_type="entity"))

    def add_section(self, heading: str, content: str, basis: str, doc_name: str) -> None:
        if not heading or not content:
            return
        self.sections.setdefault(heading, []).append(content)
        self.facts.append(Fact(label=heading, value=content, basis=basis, doc_name=doc_name, fact_type="section"))

    def add_key_value(self, key: str, value: str, basis: str, doc_name: str) -> None:
        if not key or not value:
            return
        norm_key = key.strip().lower()
        self.key_values.setdefault(norm_key, []).append(value)
        self.facts.append(Fact(label=key, value=value, basis=basis, doc_name=doc_name, fact_type="key_value"))

    def add_skill(self, skill: str, basis: str, doc_name: str) -> None:
        if skill and skill not in self.skills:
            self.skills.append(skill)
            self.facts.append(Fact(label="skill", value=skill, basis=basis, doc_name=doc_name, fact_type="skill"))

    def best_value(self, keys: List[str]) -> Optional[str]:
        for key in keys:
            values = self.key_values.get(key.lower())
            if values:
                return values[0]
        return None

    def best_section(self, keys: List[str]) -> Optional[str]:
        for key in keys:
            for heading, values in self.sections.items():
                if key.lower() in heading.lower() and values:
                    return values[0]
        return None

    def summary_text(self) -> str:
        if self.sections:
            first_section = next(iter(self.sections.values()))
            if first_section:
                return first_section[0]
        if self.key_values:
            key, values = next(iter(self.key_values.items()))
            if values:
                return f"{key}: {values[0]}"
        return ""


__all__ = ["Fact", "FactCache"]
