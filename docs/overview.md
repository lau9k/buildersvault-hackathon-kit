# Starter Kit Overview

## Why this kit exists

Most hackathons hand teams a pile of PDFs and a Slack channel. Five days later, half the teams are still wiring plumbing: loading CSVs, inventing schemas, reading API docs, arguing about what the problem even is. The weekend ends with a lot of tech demos and very few things anyone would actually use on Monday.

The BuildersVault Social Services Hackathon ships a different starter. You get a dataset that behaves like the messy reality of frontline social services work, two specific domain tracks with real operator partners, a baseline notebook that works out of the box, and a Streamlit scaffold you can extend or rip out. The goal is to get you past setup and into the interesting work within the first hour.

## What "real-world-shaped" means

The datasets in this kit are synthetic but they were designed to carry the shape of real systems, not toy data. Some specifics:

- **HIFIS 4 alignment.** Track 1's referral lifecycle, status codes, and Coordinated Access fields mirror the Homeless Individuals and Families Information System schema used by most BC social service orgs.
- **PIPA and FOIPPA consent modeling.** Consent records in Track 1 track legal basis (PIPA, FOIPPA, indigenous_governance), sharing scope, and have a state machine (active, expired, withdrawn, superseded, pending). Nine red-flag patterns are seeded with ground truth so teams can score their detectors.
- **OCAP respect.** 15% of Track 1 clients are flagged as OCAP-protected with a governing nation drawn from the Songhees, Esquimalt, Tsawout, and Pauquachin Nations. Teams building joins or ML models should honor those flags.
- **VRP-accurate routing data.** Track 2 routes, stops, and actuals match the schemas used by Onfleet and Routific. Time windows, cold chain, skill matching, and allergen safety are all enforced. Three allergen conflicts and four post-closure deliveries are seeded as ground truth for mismatch detectors.
- **Realistic messiness.** Missing DOBs at 5%, missing postal codes at 8%, phone numbers in five different formats, dates in four, null values represented four different ways. The kind of stuff every real system has and every demo carefully hides.

## Scale

The datasets are sized to fit comfortably on a laptop and to let baseline models train in under 30 seconds. Full sizes:

- Track 1: 9 organizations, 800 clients, 3,000 referrals, 10,000 service encounters, 5,000 consent records, 500 duplicate flags (300 true positive, 200 decoy false positive)
- Track 2: 2 depots, 8 vehicles, 8 drivers, 500 clients, 300 routes, 2,000 delivery requests, ~1,500 stops, 150 inventory items, ~4,900 line items

Both tracks emit Parquet (canonical), SQLite (for pre-joined views), and sample CSVs (for GitHub preview).

## What you are expected to ship Saturday

A working demo. A public repo. A short write-up of what you built and why. A statement on how your solution respects the constraints for your track. That's it. Judges will weigh problem fit, technical merit, domain grounding, and production readiness. No points for scope creep or unused libraries.

See `submission-checklist.md` for the specifics.
