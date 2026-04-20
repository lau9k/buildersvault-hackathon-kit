# Track 1: Inter-Org Referral and Care Coordination

## The Problem

Victoria has nine-plus social service organizations that each hold a partial view of every client they touch. Shelters, health outreach teams, Indigenous-led agencies, housing-first programs, mental-health and substance-use services, settlement workers. A person in crisis may interact with four of them in a single week, and each org records a different slice of that person under a slightly different name, sometimes a different date of birth, usually with a different consent form.

Referrals move between these orgs by phone call, fax, shared-inbox email, and occasional HIFIS 4 entries. Consent is fragmented across paper, PDF, and database fields that expire silently. OCAP obligations for Indigenous clients are often mishandled, not out of malice but because the systems were not designed with OCAP as a first-class primitive. Care coordination is effectively a manual process today, held together by individual caseworkers' memories and personal relationships.

One missed referral or one expired consent and a client falls through a gap. For someone at the acute end of the VI-SPDAT, that gap can be measured in weeks of lost housing progress or, in the worst case, a life.

## What the data represents

Nine mock Victoria-area orgs modeled on real HIFIS 4 deployments plus BC Coordinated Access partners. The schema draws from:

- **HIFIS 4** referral-module field names and lifecycle states.
- **BC Coordinated Access** role types (access points, matchers, housing providers).
- **AIRS taxonomy** service codes on the org side.
- **VI-SPDAT** acuity scores on clients.
- **PIPA, FOIPPA, and OCAP** structures on the consent side, including `sharing_scope_type`, `purpose_codes`, effective and expiry timestamps, and explicit `ocap_protected` flags.

Eight hundred clients, three thousand referrals, ten thousand encounters, five thousand consent records, and a small data-sharing-agreement table so you can reason about which orgs are even allowed to talk to each other.

The messiness is real. Missing DOBs, nickname variants that do not match legal names, inconsistent date formats because some orgs still export from Access, consent records whose expiries are months in the past but still being used at the encounter level, and OCAP flags that were overridden by well-meaning staff who did not understand what they were overriding. You will encounter the same data-quality issues that real HIFIS deployments see.

## Ground truth you have access to

- The forty seeded near-duplicate client groupings are documented in `duplicate_flags.csv` with 300 true-positive pairs and 200 decoy false-positive pairs. Use these to compute precision, recall, and F1 for your entity-resolution work.
- The nine consent red-flag patterns are tagged in the `notes` column on affected rows with `RED_FLAG_` prefixes, for example `RED_FLAG_EXPIRED_CONSENT_USED`, `RED_FLAG_OCAP_OVERRIDE`, `RED_FLAG_SCOPE_MISMATCH`. Surface these in your solution and you have a defensible evaluation story.

## Four concrete challenge angles

Pick one. Going deep on one beats being shallow on all four.

1. **Entity resolution.** Merge duplicate client records across agencies without violating OCAP. Beat the notebook baseline on F1. Bonus for explainable match decisions.
2. **Consent-gap surfacing.** Detect encounters and data flows that happen under expired, withdrawn, or scope-mismatched consent. Build a view a caseworker can trust at the point of care.
3. **Referral lifecycle analytics.** Find the stages where referrals die. Quantify time-in-state, decline reasons, and receiving-org capacity constraints. Recommend where coordinators should intervene.
4. **Falling-through-the-gaps risk detection.** Identify clients who are chronically vulnerable, have lost recent contact, have expired consent, and have a stalled referral. Score and rank. Explain your features.

## Privacy constraints your solution must respect

These are non-negotiable. Judges will check.

1. **OCAP-protected client data cannot be shared beyond the governing nation's agreed scope.** If `ocap_protected = true`, your pipeline must not expose that client's data to orgs outside the explicitly listed OCAP-approved partners, regardless of legitimate care-coordination intent.
2. **Withdrawn consent means no new data use.** Not even for care coordination, not even for analytics. A client with `consent.status = withdrawn` disappears from downstream consumers the moment withdrawal takes effect.
3. **FOIPPA statutory sharing requires purpose_codes to be populated.** If `purpose_codes` is null or empty, the share is unlawful. Your code must refuse to emit that record.
4. **`sharing_scope_type = single_agency_only` means no multi-agency joins at query time.** If a client's current consent is single-agency, your system must not join their data with any other org's records, even for the golden-view join pattern shown in the notebook.

Build these as first-class rules. Mention them in your README. Show the judges where they are enforced in code.

## Suggested tech stack

- **pandas** or **duckdb** for analytics. Duckdb handles the golden join and consent-aware filters in one SQL statement if you prefer SQL over Python.
- **Streamlit** or **Next.js** for dashboarding.
- **Splink** or **recordlinkage** for entity resolution, sentence-transformers for embedding-based name matching.
- **Personize SDK** (optional) if you want to layer memory and governance on top. API-HTTP, not MCP.
- **pytest** with a handful of tests that prove your privacy gates actually reject bad inputs.

## Where to go from here

- Open `../notebooks/00_quickstart.ipynb` and run it end to end. It will give you the golden join plus a baseline to beat.
- Read `../docs/erd.mmd` for the full data model.
- Read `../dictionary/fields.csv` for exact field semantics, legal sources, and valid enumerations.
- Check `../README.md` for submission rules and evaluation rubric.

Good luck. Build something that a caseworker would actually trust.
