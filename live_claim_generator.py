"""Generate claim payloads without connecting to PostgreSQL.

The database consumer lives in agentfor.py. This module only produces data.
"""

from __future__ import annotations

import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


COUNTIES = [
    "NAIROBI", "KIAMBU", "MOMBASA", "KISUMU", "NAKURU",
    "MACHAKOS", "UASIN GISHU", "KAKAMEGA", "BUNGOMA", "NAROK",
]
INSURERS = ["SHA", "JUBILEE", "AAR", "BRITAM", "CIC"]
FACILITIES = [
    ("NAIROBI_HOSPITAL", "Nairobi Hospital", "NAIROBI", "level 5", "PRIVATE"),
    ("KENYATTA_NATIONAL", "Kenyatta National Hospital", "NAIROBI", "level 6", "PUBLIC"),
    ("KIAMBU_COUNTY", "Kiambu County Hospital", "KIAMBU", "level 4", "PUBLIC"),
    ("MOMBASA_REFERRAL", "Mombasa County Referral", "MOMBASA", "level 5", "PUBLIC"),
    ("NAKURU_LEVEL_5", "Nakuru Level 5 Hospital", "NAKURU", "level 5", "PUBLIC"),
]
CLAIM_TYPES = ["INSTITUTIONAL", "OUTPATIENT", "MATERNITY"]
CLAIM_SUBTYPES = ["OP", "IP", "MATERNITY", "DENTAL", "OPTICAL"]


def generate_claim() -> dict[str, Any]:
    """Produce one database-shaped claim without performing I/O."""

    facility, facility_name, county, level, category = random.choice(FACILITIES)
    now = datetime.now(timezone.utc)
    date_start = now.date() - timedelta(days=random.randint(0, 30))
    return {
        "claim_id": str(uuid.uuid4()),
        "creation": now,
        "modified": now,
        "bundle_id": str(uuid.uuid4()),
        "claim_status": random.choice(["CREATED", "RECEIVED", "QUEUED"]),
        "claim_type": random.choice(CLAIM_TYPES),
        "claim_subtype": random.choice(CLAIM_SUBTYPES),
        "claim_amount": round(random.triangular(0, 150000, 12000), 2),
        "date_start": date_start,
        "date_end": date_start + timedelta(days=random.randint(0, 5)),
        "client": f"CLIENT-{random.randint(1, 500):04d}",
        "facility": facility,
        "facility_name": facility_name,
        "facility_level": level,
        "facility_category": category,
        "county": county,
        "sub_county": county,
        "scheme_id": random.choice(["GENERAL", "CORPORATE", "FAMILY"]),
        "insurer": random.choice(INSURERS),
        "claim_use": "HEALTHCARE",
        "is_cancelled": random.random() < 0.04,
        "diagnoses_code": random.choice(["J06.9", "I10", "E11.9", "A09"]),
        "interventions_code": random.choice(["CONSULTATION", "LAB_TEST", "XRAY", "SURGERY"]),
        "sub_interventions_code": random.choice(["CONSULTATION", "MEDICATION", "PHYSIOTHERAPY"]),
        "linked_claim": None,
        "claim_upstream_error_group": None,
        "claim_upstream_errors": None,
        "claim_verdict": "PENDING",
        "dbt_loaded_at": now,
    }


def generate_claims(count: int = 1) -> list[dict[str, Any]]:
    """Produce a bounded batch of database-shaped claim payloads."""

    return [generate_claim() for _ in range(max(1, int(count)))]


def run_live_stream(
    consumer: Callable[[dict[str, Any]], Any] | None = None,
    interval_seconds: float = 10.0,
    batch_min: int = 1,
    batch_max: int = 5,
) -> None:
    """Produce live claims and optionally pass each payload to a consumer."""

    print("Live claim producer started; it does not connect to PostgreSQL.")
    try:
        while True:
            claims = generate_claims(random.randint(batch_min, batch_max))
            if consumer is not None:
                for claim in claims:
                    consumer(claim)
            print(f"Produced {len(claims)} claim payloads")
            time.sleep(max(0.1, interval_seconds))
    except KeyboardInterrupt:
        print("Live claim producer stopped.")


if __name__ == "__main__":
    run_live_stream()
