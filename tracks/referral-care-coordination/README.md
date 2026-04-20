# Track 1: Inter-Org Referral & Care Coordination

Victoria has 9+ social service organizations that each hold a partial view of the same client. Referrals move between them by email, phone, and paper. Consent is tracked inconsistently. OCAP obligations for Indigenous clients are often honored informally. Your job: build a tool that makes coordination safer, faster, or more visible, without violating the privacy and sovereignty constraints that govern this data.

## Quickstart

```bash
# From repo root
pip install -r requirements.txt
python tracks/referral-care-coordination/generator/generate.py
jupyter lab tracks/referral-care-coordination/notebooks/00_quickstart.ipynb
```

The generator runs in about 30 seconds and produces:

- 7 Parquet files in `data/raw/`
- 1 SQLite database in `data/raw/track1.sqlite` with two pre-joined views: `v_referrals_enriched` and `v_client_timeline`
- 7 sample CSVs (first 1000 rows each) in `data/sample/`

## Data at a glance

| Table | Rows | What it is |
|---|---|---|
| organizations | 9 | Mock Victoria agencies (shelter, outreach, addictions, mental health, legal aid, food bank, housing) |
| clients | 800 | People who interact with one or more of the orgs |
| referrals | 3,000 | Full-lifecycle referral events between orgs |
| service_encounters | 10,000 | Shelter stays, outreach contacts, office visits |
| consent_records | 5,000 | PIPA / FOIPPA / OCAP consent with state machine |
| data_sharing_agreements | 4 | Seed DSAs referenced by consent_records |
| duplicate_flags | ~500 | 300 true positive + 200 decoy pairs |

See `dictionary/fields.csv` for every column.

## Domain anchors

- HIFIS 4 referral module
- BC Coordinated Access (By-Name List, priority levels)
- AIRS / 211 service taxonomy codes
- VI-SPDAT assessment scoring (0-17)
- PIPA (Personal Information Protection Act, BC)
- FOIPPA (Freedom of Information and Protection of Privacy Act, BC)
- OCAP (Ownership, Control, Access, Possession — First Nations data sovereignty)

Read `docs/problem-framing.md` for how the kit translates these into data.

## Golden join

The one query every team writes on day one is the `referrals ⋈ clients ⋈ referring_org ⋈ receiving_org ⋈ current_consent` join. The starter notebook constructs it in cell 6 as `referrals_enriched`. The SQLite file exposes it as `v_referrals_enriched`. Start there.

## Seeded ground truth (for scoring your solutions)

- **40 near-duplicate client pairs** across agencies. Documented in `duplicate_flags.csv` as true positives (match_score >= 0.75).
- **200 decoy false positives** in `duplicate_flags.csv` (match_score 0.55-0.80). A good dedup detector rejects these.
- **9 consent red-flag patterns** seeded in `consent_records.notes` with prefix `RED_FLAG:`. Specifics in `docs/problem-framing.md`.

Your solution can self-score against these.

## Constraints your solution must respect

1. **OCAP-protected clients.** 15% of clients are flagged `ocap_protected=True`. Their data cannot be shared beyond the governing nation's agreed scope. If your solution exports or joins these clients with unverified consent, that's a fail.
2. **Withdrawn consent.** `consent.status='withdrawn'` means no new data use after `withdrawal_date`. A solution that keeps displaying withdrawn-consent records in a coordination UI is broken.
3. **Sharing scope.** `consent.sharing_scope_type='single_agency_only'` means that client's data cannot appear in a multi-agency view at query time.
4. **FOIPPA statutory sharing.** Requires `purpose_codes` to be populated. Empty `purpose_codes` on a FOIPPA record is a governance gap.

Judges will check at least one of these.

## Baseline that ships

The starter notebook includes a rule-based duplicate detector (Soundex blocking + SequenceMatcher string similarity + DOB match + address proxy) that hits roughly 60-70% recall at 80% precision against the seeded ground truth. Your team should be able to beat this. Candidates: Splink, recordlinkage, embedding-based name matching, rules + learned hybrid.

## Extension ideas

- Entity resolution pipeline that honors OCAP (merge within nation, flag across)
- Consent-gap surfacing dashboard (active encounters without matching active consent)
- Referral lifecycle analytics (where do referrals die? what predicts decline?)
- Risk detection for chronic homeless + lost contact + expired consent
- Care coordination API for an AI caseworker agent

See `docs/problem-framing.md` for the full treatment.
