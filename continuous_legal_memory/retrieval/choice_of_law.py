"""
Cross-Jurisdictional Conflict of Laws (Private International Law / Choice-of-Law Router).

Implements formal choice-of-law routing grounded in Private International Law:
- Rome I Regulation (EC) No 593/2008 Articles 3 (Freedom of Choice) & 9 (Overriding Mandatory
  Provisions / *Lois de Police*)
- GDPR Article 3(2) & Brazil LGPD Article 3 (Extraterritorial Data Subject Protection /
  *Lex Loci Protectionis*)
- Restatement (Second) of Conflict of Laws § 187–188 (*Lex Loci Solutionis*)

Resolves multi-sovereign normative conflicts when a transaction spans multiple jurisdictions
(e.g., an EU data subject processed by a Brazilian subsidiary under a US-NY commercial contract),
ensuring that overriding mandatory statutes (*lois de police*) of the protected domicile or
place of performance prevail over contractual choice-of-forum clauses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from continuous_legal_memory.domain.models import MemoryRecord

SUPPORTED_JURISDICTION_REGIMES: dict[str, dict[str, str]] = {
    "EU": {
        "name": "European Union",
        "extraterritorial_statute": "GDPR Art. 3(2) & EU AI Act Art. 2(1)",
        "conflict_of_laws_rule": "Rome I Regulation (EC) 593/2008 Art. 3 & Art. 9 (Lois de Police)",
    },
    "US": {
        "name": "United States",
        "extraterritorial_statute": "CCPA/CPRA Cal. Civ. Code § 1798.140 & HIPAA 45 CFR § 160.102",
        "conflict_of_laws_rule": "Restatement (Second) of Conflict of Laws § 187–188",
    },
    "BR": {
        "name": "Brazil",
        "extraterritorial_statute": "LGPD Lei 13.709/2018 Art. 3",
        "conflict_of_laws_rule": "LINDB Decreto-Lei 4.657/1942 Art. 9 & Art. 17 (Ordem Pública)",
    },
    "UK": {
        "name": "United Kingdom",
        "extraterritorial_statute": "UK GDPR Art. 3(2) & Data Protection Act 2018 s. 207",
        "conflict_of_laws_rule": "Retained Rome I Regulation (UK) Art. 9 & FCA SYSC Territorial Scope",
    },
    "SG": {
        "name": "Singapore",
        "extraterritorial_statute": "Singapore PDPA 2012 s. 4 & MAS TRM Guidelines",
        "conflict_of_laws_rule": "Common Law Proper Law Doctrine & PDPA Mandatory Baseline",
    },
    "JP": {
        "name": "Japan",
        "extraterritorial_statute": "Japan APPI Act No. 57 of 2003 Art. 171 (Extraterritorial Application)",
        "conflict_of_laws_rule": "Act on General Rules for Application of Laws (Tsusokuho) Art. 7 & Art. 42",
    },
    "CH": {
        "name": "Switzerland",
        "extraterritorial_statute": "Swiss FADP / revDSG (SR 235.1) Art. 3(2) (Effects Doctrine)",
        "conflict_of_laws_rule": "Swiss PILA / IPRG (SR 291) Art. 18 (Lois d'application immédiate)",
    },
}


def _base_jurisdiction(code: str | None) -> str | None:
    if not code:
        return None
    return code.strip().upper().split("-")[0]


@dataclass
class ConnectingFactors:
    """
    Private International Law connecting factors (*elementos de conexão*) for a query or transaction.

    Attributes:
        data_subject_domicile: Jurisdiction of the data subject or protected party (*lex loci protectionis*).
        place_of_performance: Jurisdiction where processing or contract execution occurs (*lex loci solutionis*).
        choice_of_forum: Contractual choice-of-law jurisdiction designated by party autonomy (*lex voluntatis*).
        lex_fori: Jurisdiction of the supervising regulator or forum court.
    """

    data_subject_domicile: str | None = None
    place_of_performance: str | None = None
    choice_of_forum: str | None = None
    lex_fori: str | None = None

    @classmethod
    def from_input(cls, data: ConnectingFactors | dict[str, Any] | None) -> ConnectingFactors | None:
        if data is None:
            return None
        if isinstance(data, ConnectingFactors):
            return data
        return cls(
            data_subject_domicile=data.get("data_subject_domicile"),
            place_of_performance=data.get("place_of_performance"),
            choice_of_forum=data.get("choice_of_forum"),
            lex_fori=data.get("lex_fori"),
        )


@dataclass
class ChoiceOfLawResolution:
    """
    Result of choice-of-law analysis over a set of `ConnectingFactors`.
    """

    applicable_jurisdictions: list[str] = field(default_factory=list)
    primary_jurisdiction: str | None = None
    overriding_mandatory_jurisdictions: list[str] = field(default_factory=list)
    contractual_forum_jurisdiction: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable_jurisdictions": self.applicable_jurisdictions,
            "primary_jurisdiction": self.primary_jurisdiction,
            "overriding_mandatory_jurisdictions": self.overriding_mandatory_jurisdictions,
            "contractual_forum_jurisdiction": self.contractual_forum_jurisdiction,
            "rationale": self.rationale,
        }


def is_overriding_mandatory_provision(
    record: MemoryRecord,
    mandatory_authority_threshold: int = 5,
) -> bool:
    """
    Determine whether a `MemoryRecord` qualifies as an Overriding Mandatory Provision
    (*Loi de Police* / *Norma de Ordem Pública Internacional*) that cannot be derogated
    by contractual choice of law (Rome I Art. 9).
    """
    if isinstance(record.metadata, dict):
        if record.metadata.get("overriding_mandatory") is True:
            return True
        if record.metadata.get("loi_de_police") is True:
            return True
        if record.metadata.get("public_order") is True:
            return True
    return int(record.authority_rank) >= mandatory_authority_threshold


class ChoiceOfLawRouter:
    """
    Private International Law router that filters and weights multi-jurisdictional
    `MemoryRecord` candidates according to statutory connecting factors.
    """

    def __init__(self, mandatory_authority_threshold: int = 5) -> None:
        self.mandatory_authority_threshold = int(mandatory_authority_threshold)

    def resolve_factors(
        self,
        factors: ConnectingFactors | dict[str, Any],
    ) -> ChoiceOfLawResolution:
        """
        Determine ordered applicable jurisdictions and overriding mandatory jurisdictions
        from `ConnectingFactors`.
        """
        cf = ConnectingFactors.from_input(factors) or ConnectingFactors()
        mandatory_juris: list[str] = []
        applicable: list[str] = []

        for j in (cf.data_subject_domicile, cf.place_of_performance, cf.lex_fori):
            if j and j not in mandatory_juris:
                mandatory_juris.append(j)
            if j and j not in applicable:
                applicable.append(j)

        if cf.choice_of_forum and cf.choice_of_forum not in applicable:
            applicable.append(cf.choice_of_forum)

        primary = (
            cf.data_subject_domicile
            or cf.place_of_performance
            or cf.choice_of_forum
            or cf.lex_fori
        )

        rationale_parts: list[str] = []
        if cf.data_subject_domicile:
            base_dom = _base_jurisdiction(cf.data_subject_domicile)
            statute_ref = (
                SUPPORTED_JURISDICTION_REGIMES[base_dom]["extraterritorial_statute"]
                if base_dom in SUPPORTED_JURISDICTION_REGIMES
                else "overriding mandatory protection"
            )
            rationale_parts.append(
                f"lex_loci_protectionis={cf.data_subject_domicile} ({statute_ref})"
            )
        if cf.place_of_performance:
            rationale_parts.append(f"lex_loci_solutionis={cf.place_of_performance}")
        if cf.choice_of_forum:
            rationale_parts.append(f"lex_voluntatis={cf.choice_of_forum} (dispositive contract forum)")
        if cf.lex_fori:
            rationale_parts.append(f"lex_fori={cf.lex_fori}")

        return ChoiceOfLawResolution(
            applicable_jurisdictions=applicable,
            primary_jurisdiction=primary,
            overriding_mandatory_jurisdictions=mandatory_juris,
            contractual_forum_jurisdiction=cf.choice_of_forum,
            rationale="; ".join(rationale_parts) if rationale_parts else "unrestricted_jurisdiction",
        )

    def route_candidates(
        self,
        valid_indices: list[int],
        valid_records: list[MemoryRecord],
        jurisdiction: str | list[str] | None = None,
        connecting_factors: ConnectingFactors | dict[str, Any] | None = None,
    ) -> tuple[list[int], list[MemoryRecord], list[float], ChoiceOfLawResolution | None]:
        """
        Filter `(valid_indices, valid_records)` by jurisdictional scope and compute
        Private International Law precedence adjustments.

        Returns:
            Tuple of `(routed_indices, routed_records, choice_of_law_bonuses, resolution)`.
        """
        if not valid_indices or not valid_records:
            return valid_indices, valid_records, [], None

        # 1. Explicit jurisdiction filter (single or list)
        if jurisdiction is not None and connecting_factors is None:
            allowed = {jurisdiction} if isinstance(jurisdiction, str) else set(jurisdiction)
            exact_pairs = [
                (idx, rec)
                for idx, rec in zip(valid_indices, valid_records)
                if rec.jurisdiction in allowed
            ]
            if not exact_pairs:
                # Fallback to universal (jurisdiction=None) records if present
                exact_pairs = [
                    (idx, rec)
                    for idx, rec in zip(valid_indices, valid_records)
                    if rec.jurisdiction is None
                ]
            routed_idx = [p[0] for p in exact_pairs]
            routed_rec = [p[1] for p in exact_pairs]
            res = ChoiceOfLawResolution(
                applicable_jurisdictions=sorted(allowed),
                primary_jurisdiction=next(iter(allowed)) if allowed else None,
                rationale=f"explicit_jurisdiction_filter={sorted(allowed)}",
            )
            return routed_idx, routed_rec, [0.0] * len(routed_rec), res

        # 2. Private International Law ConnectingFactors routing
        if connecting_factors is not None:
            cf = ConnectingFactors.from_input(connecting_factors) or ConnectingFactors()
            resolution = self.resolve_factors(cf)
            allowed_set = set(resolution.applicable_jurisdictions)

            filtered_pairs = [
                (idx, rec)
                for idx, rec in zip(valid_indices, valid_records)
                if rec.jurisdiction in allowed_set or rec.jurisdiction is None
            ]
            if not filtered_pairs:
                return [], [], [], resolution

            # Check whether any candidate in the pool is an overriding mandatory provision
            # from the domicile / performance / forum jurisdiction
            has_mandatory_override = any(
                rec.jurisdiction in resolution.overriding_mandatory_jurisdictions
                and is_overriding_mandatory_provision(rec, self.mandatory_authority_threshold)
                for _, rec in filtered_pairs
            )

            routed_idx = []
            routed_rec = []
            bonuses: list[float] = []

            for idx, rec in filtered_pairs:
                routed_idx.append(idx)
                routed_rec.append(rec)
                is_mand = is_overriding_mandatory_provision(rec, self.mandatory_authority_threshold)

                if has_mandatory_override:
                    # Overriding mandatory provision (Rome I Art. 9 / GDPR Art. 3(2) / LGPD Art. 3)
                    # prevails over contractual choice_of_forum
                    if rec.jurisdiction == cf.data_subject_domicile and is_mand:
                        bonuses.append(25.0)
                    elif rec.jurisdiction in resolution.overriding_mandatory_jurisdictions and is_mand:
                        bonuses.append(18.0)
                    elif rec.jurisdiction == cf.choice_of_forum and not is_mand:
                        bonuses.append(-15.0)
                    else:
                        bonuses.append(0.0)
                else:
                    # Dispositive commercial matter: party autonomy (choice_of_forum) prevails
                    if cf.choice_of_forum and rec.jurisdiction == cf.choice_of_forum:
                        bonuses.append(15.0)
                    elif cf.place_of_performance and rec.jurisdiction == cf.place_of_performance:
                        bonuses.append(8.0)
                    elif cf.data_subject_domicile and rec.jurisdiction == cf.data_subject_domicile:
                        bonuses.append(5.0)
                    else:
                        bonuses.append(0.0)

            return routed_idx, routed_rec, bonuses, resolution

        return valid_indices, valid_records, [0.0] * len(valid_records), None
