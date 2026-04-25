# BuildersVault Social Services Hackathon — Starter Kit

**Event:** BuildersVault Social Services Hackathon
**Kickoff:** Monday April 20, 2026, 5:30 PM at Modus Design Labs (Owen Works), Victoria BC
**Demo & Awards:** Saturday April 25, 2026, 6:30 to 9:30 PM at UVic Bob Wright B150 Flury Hall
**Register:** https://luma.com/uvqu2y5o

---

## Hackathon Goal

Build AI tools that frontline Victoria social services teams can actually use. You have five days and two tracks. Each track ships with real-world-shaped synthetic data, a domain primer, a baseline notebook, and a Streamlit scaffold so you can spend the weekend building value instead of wiring plumbing.

**Judging themes:**

1. **Problem Fit** — does this solve a real operator pain?
2. **Technical Merit** — is the approach sound?
3. **Social Services Domain Grounding** — does it respect consent, safety, and dignity?
4. **Production Readiness** — could a frontline worker run this Monday morning?

---

## What's in the Kit

```
hackathon-starter-kit/
├── tracks/
│   ├── referral-care-coordination/    # Track 1
│   └── food-security-delivery/        # Track 2
├── shared/
│   ├── src/                           # loaders, validators, messiness helpers
│   └── app/                           # Streamlit multi-page explorer
├── docs/                              # overview, submission checklist
├── README.md, LICENSE, DATA_LICENSE.md, requirements.txt
```

Each track folder contains:

- `generator/generate.py` — deterministic Python generator (seed=42) that produces the full dataset locally
- `data/raw/*.parquet` + `track?.sqlite` — canonical files with pre-joined views
- `data/sample/*.csv` — small CSV samples for quick preview (under GitHub render limits)
- `dictionary/fields.csv` — one row per column with type, description, enum values, FK target, example
- `docs/erd.mmd` — Mermaid ER diagram, renders natively in GitHub
- `docs/problem-framing.md` — what the data represents, seeded ground truth, 4 challenge angles, constraints your solution must respect
- `notebooks/00_quickstart.ipynb` — 10-cell tour that loads the data, validates the schema, runs the golden join, plots EDA, and ships a baseline model

---

## Quickstart (3 commands)

```bash
# 1. Clone + install
git clone https://github.com/<org>/hackathon-starter-kit.git
cd hackathon-starter-kit
pip install -r requirements.txt

# 2. Generate the data for the track you care about (or both)
python tracks/referral-care-coordination/generator/generate.py
python tracks/food-security-delivery/generator/generate.py

# 3. Open the quickstart notebook
jupyter lab tracks/referral-care-coordination/notebooks/00_quickstart.ipynb
```

Then explore visually:

```bash
streamlit run shared/app/streamlit_app.py
```

---

## Tracks

### Track 1: Inter-Org Referral & Care Coordination

Victoria has 9+ social service orgs (shelters, outreach, addictions, mental health, legal aid, food bank, housing). Each holds a partial view of the same client. Referrals are ad-hoc. Consent is fragmented. OCAP obligations for Indigenous clients are often mishandled. Your job: build something that makes coordination safer, faster, or more visible.

Domain anchors: HIFIS 4 referral module, BC Coordinated Access, AIRS service taxonomy, VI-SPDAT assessment, PIPA/FOIPPA/OCAP.

**Data:** 9 orgs, 800 clients, 3,000 referrals, 10,000 service encounters, 5,000 consent records, 500 duplicate flags (300 true positive, 200 decoy false positive).

See `tracks/referral-care-coordination/README.md` for track-specific details.

### Track 2: Food Security Delivery Operations

Victoria has a growing population of mobility-limited residents who rely on delivered meals and grocery hampers. Operators are mostly volunteer-led. Route planning is manual. Allergen and mobility matching is tribal knowledge. Missed deliveries mean missed meals. Your job: build something that optimizes the run, flags mismatches, or reduces volunteer burnout.

Domain anchors: Meals on Wheels, grocery-hamper delivery, vehicle routing (Onfleet / Routific schemas), cold-chain handling, volunteer ops.

**Data:** 2 depots, 8 vehicles (5 refrigerated, 1 wheelchair-lift), 8 drivers (6 volunteer), 500 clients, 10,000 delivery requests, 300 routes over 3 weeks, ~3,500 route stops with actuals (realistic MOW density at roughly 17 stops per active route), 150 inventory items, ~24,000 line items.

See `tracks/food-security-delivery/README.md` for track-specific details.

---

## First 30 Minutes

The fastest path from clone to "I understand the data":

1. Clone, install requirements, pick a track.
2. Run `python generate.py` (takes ~30 seconds).
3. Open `notebooks/00_quickstart.ipynb`, run every cell. Stop and read cells 5 (schema validation) and 6 (golden join).
4. Skim `docs/problem-framing.md` and `docs/erd.mmd`.
5. Grep `fields.csv` for any column name you don't recognize.
6. Open the Streamlit app and click through Explore / Baseline / Map.

After that you will know more about the dataset than most teams will learn all weekend.

---

## Repo Map

Top-level:

- `README.md` (this file)
- `LICENSE` — MIT for code
- `DATA_LICENSE.md` — CC BY 4.0 for synthetic data
- `requirements.txt` — Python deps for generators and notebooks
- `.gitignore`
- `docs/overview.md` — longer context on the hackathon
- `docs/submission-checklist.md` — what judges expect Saturday night

Track folders are identical in shape so muscle memory transfers between tracks.

Shared `shared/src/` is imported by both generators and both notebooks. Shared `shared/app/` is a Streamlit multi-page scaffold. Extend it or rip it out.

---

## ERD & Golden Join Path

Each track has a Mermaid ERD at `tracks/<track>/docs/erd.mmd`. GitHub renders it natively. The "golden join" in each notebook (cell 6) gives you the one query every team ends up writing on day one. Start there.

---

## Submission & Demo Checklist

See `docs/submission-checklist.md` for the full list. At a minimum Saturday:

- Working demo (live or recorded, 3 minutes max)
- Public GitHub repo with your code
- One-paragraph README explaining what you built and why
- A statement on how your solution respects privacy and safety constraints for the track

---

## Licensing & IP

Code in this kit is MIT. The synthetic data is CC BY 4.0. Everything you build during the weekend is yours. BuildersVault does not claim IP on team work. Partners (Xenex Consultin Inc. / FatehCare, and others involved as speakers or judges) may express interest in seeing solutions after the weekend. That conversation is between you and them.

No real personally identifiable information appears in any file in this repo. Every name, address, phone number, and ID is generated with Faker using seed=42.

---

## Support

- Event community channel: https://www.linkedin.com/posts/buildersvault_the-repo-is-a-scaffold-the-theme-is-the-activity-7452244679225683968-CTHP?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAACQx0FYBScC4YxFx44SDSoSuwijABPebMqo
- Starter kit issues: open a GitHub issue on this repo
- Data or generator bugs: tag the issue `dataset`
- Questions during the event: ping in the comments on LinkedIn. Let's build in public
- Event host: Lautaro Cepeda (BuildersVault)
