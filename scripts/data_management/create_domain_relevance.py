#!/usr/bin/env python3
"""
Generate domain_relevance.csv from metadata across all data types.
Maps (domain_from, domain_to) -> relevance for dataset-to-dataset relevance weighting.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(DATA_ROOT, "shared", "domain_relevance.csv")

# User-selectable domains (from web_app / API)
USER_DOMAINS = [
    "general", "healthcare", "finance", "cybersecurity",
    "autonomous_vehicles", "recommendation_systems", "manufacturing", "iot"
]

# Known domain relationships: (domain_from, domain_to) -> relevance
# Same domain = 1.0, related = 0.7-0.9, unrelated = 0.3
DOMAIN_RELATED_GROUPS = {
    "healthcare": ["health and medicine", "healthcare", "health", "medicine", "biology"],
    "finance": ["finance", "financial", "business"],
    "cybersecurity": ["cybersecurity", "computer science", "security"],
    "manufacturing": ["manufacturing", "industry", "physics and chemistry"],
    "iot": ["iot", "iot / sensors", "climate and environment", "computer science"],
    "autonomous_vehicles": ["autonomous_vehicles", "autonomous vehicles", "physics and chemistry", "computer science"],
    "recommendation_systems": ["recommendation_systems", "recommendation systems", "social science", "business"],
}


def _normalize_domain(s: str) -> str:
    """Normalize domain string for lookup."""
    if not s or (hasattr(pd, "isna") and pd.isna(s)):
        return ""
    return str(s).strip().lower()


# Domains observed in metadata (from data/tabular, image, text, timeseries)
METADATA_DOMAINS = [
    "biology", "health and medicine", "social science", "physics and chemistry",
    "climate and environment", "games", "other", "computer science", "business",
]

def collect_domains_from_metadata() -> set:
    """Extract unique domains from metadata; fallback to known list if read fails."""
    domains = set(METADATA_DOMAINS)
    for dtype in ["tabular", "image", "text", "timeseries"]:
        meta_path = os.path.join(DATA_ROOT, dtype, "dataset_metadata.csv")
        if not os.path.exists(meta_path):
            continue
        try:
            df = pd.read_csv(meta_path, usecols=["domain"], low_memory=False)
            for d in df["domain"].dropna().astype(str).str.strip().str.lower():
                if d and d != "nan":
                    domains.add(d)
        except Exception as e:
            print(f"Warning: Could not read {meta_path}: {e}")
    return domains


def build_relevance_matrix(metadata_domains: set) -> list[tuple[str, str, float]]:
    """Build (domain_from, domain_to, relevance) rows."""
    rows = []
    all_domains = set(metadata_domains) | set(USER_DOMAINS)

    def add(from_d: str, to_d: str, rel: float):
        from_n = _normalize_domain(from_d)
        to_n = _normalize_domain(to_d)
        if from_n and to_n:
            rows.append((from_n, to_n, rel))

    # Same domain = 1.0
    for d in all_domains:
        dn = _normalize_domain(d)
        if dn:
            add(d, d, 1.0)

    # Map metadata domains to user domains
    for user_d in USER_DOMAINS:
        user_n = _normalize_domain(user_d)
        if not user_n:
            continue
        # Find which group this user domain belongs to
        for group_name, members in DOMAIN_RELATED_GROUPS.items():
            if user_n in [m.lower() for m in members]:
                for member in members:
                    add(member, user_d, 1.0)
                    add(user_d, member, 1.0)
                break

    # Cross-group relevance: healthcare <-> biology, finance <-> business, etc.
    for group_name, members in DOMAIN_RELATED_GROUPS.items():
        members_lower = [m.lower() for m in members]
        for i, m1 in enumerate(members):
            for m2 in members[i + 1 :]:
                add(m1, m2, 0.85)
                add(m2, m1, 0.85)

    # general: 0.5 for all other domains
    for d in all_domains:
        dn = _normalize_domain(d)
        if dn and dn != "general":
            add("general", dn, 0.5)
            add(dn, "general", 0.5)

    # Default unrelated: 0.3 for any pair not yet covered
    seen = {(r[0], r[1]) for r in rows}
    for d1 in all_domains:
        for d2 in all_domains:
            k1, k2 = _normalize_domain(d1), _normalize_domain(d2)
            if k1 and k2 and (k1, k2) not in seen:
                rows.append((k1, k2, 0.3))
                seen.add((k1, k2))

    return rows


def main() -> int:
    metadata_domains = collect_domains_from_metadata()
    print(f"Found {len(metadata_domains)} unique domains from metadata: {sorted(metadata_domains)[:20]}...")

    rows = build_relevance_matrix(metadata_domains)
    df = pd.DataFrame(rows, columns=["domain_from", "domain_to", "relevance"])
    df = df.drop_duplicates(subset=["domain_from", "domain_to"], keep="first")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
