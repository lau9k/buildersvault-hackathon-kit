# Track 2: Food Security Delivery Operations

## The Problem

Victoria has a growing number of mobility-limited residents who depend on food being delivered to their door. Meals on Wheels, grocery hamper programs, and cultural-specific meal services together feed hundreds of clients every week. The operators running these routes are mostly volunteers. Route planning is done by hand on a whiteboard or in a spreadsheet. Dietary and allergy matching lives in the head of whichever long-time volunteer happens to be on shift. Volunteer burnout and no-show rates are not systematically tracked, so the ops lead only finds out someone is done when they stop showing up.

Two things make this high-stakes. A missed delivery is a client missing a meal that day, and for food-insecure clients there is often no plan B. An allergen-matched delivery is, in the worst case, a hospitalization. The margin for "good enough" is thin.

## What the data represents

You get 500 mock Victoria clients modeled on a blended Meals on Wheels plus hamper operation. The data captures the full operational picture:

- **Clients**: address, mobility level, wheelchair use, two-person-lift requirement, primary language, dietary restrictions, allergens with severity, enrolment status, food security level, home depot.
- **Drivers**: 8 total, 6 volunteer and 2 paid staff, with skill tags (lift-trained, two-person-trained, first-aid certified), language fluency, max hours and distance per week.
- **Vehicles**: 8 total, 1 with wheelchair lift, 3 refrigerated, 4 standard, each with capacity and cold-chain specs.
- **Depots**: 2 anchors, a central kitchen and a hamper warehouse.
- **Routes**: 300 routes across 3 weeks with planned versus actual stops, distances, durations, and on-time rates.
- **Delivery requests**: 2,000 total, each with a time window, program tag (MOW, hamper, cultural), and dietary/mobility snapshot captured at request time.
- **Route stops**: roughly 10,000, with arrival times, status, failure reasons, and driver notes.
- **Inventory items**: 150, tagged with allergens, dietary flags, and cold-chain requirements.
- **Delivery request items**: the bridge table showing what items were actually loaded for each request.

The data respects real operational constraints: time windows, vehicle capacity, cold-chain integrity, skill-to-need matching, allergen safety. It was not generated from a uniform random distribution. It was generated from the constraint graph an actual ops lead works within.

## Ground truth seeded for scoring

So that your mismatch detector has something to catch, we have deliberately seeded the data with known issues:

- **3 allergen conflict violations** in `delivery_request_items` (a severe-allergen item delivered to a client with that allergen). These are tagged `DATA_QUALITY_ISSUE` in the corresponding request `notes` field.
- **4 post-closure delivery attempts** (a stop attempted at a client whose `enrolment_status` was `closed` or `deceased` at the time of the route). These are tagged in the stop's `driver_notes`.
- **5% of two-person-required clients** have at least one stop marked with failure reason `requires_two_person_unavailable`, reflecting a driver arriving solo.

Use these for scoring your mismatch detector. If your solution finds these seeded issues plus additional plausible ones, you are on the right track. If it misses them, you have a precision problem worth fixing before demo day.

## Four concrete challenge angles

1. **Route optimization**: shrink total drive time and raise on-time rate while honoring time windows, vehicle capacity, cold-chain, and skill matching.
2. **Compatibility mismatch detection**: surface wheelchair clients on non-lift vehicles, allergen-item conflicts, driver-client language gaps, pet allergy conflicts.
3. **Volunteer retention analytics**: predict no-show risk per driver per week, flag early burnout signals (rising no-shows, shrinking hours, worsening on-time rate), measure fairness of route assignment.
4. **Demand forecasting**: predict request volume by program type and depot for the next 1 to 4 weeks so the ops lead can pre-stage inventory.

Pick one. Go deep. Judges reward a shipped demo on one angle far more than a surface-level attempt at all four.

## Operational constraints your solution must respect

- **Consent and enrolment**: clients with `enrolment_status` of `closed` or `deceased` cannot receive deliveries. No exceptions.
- **Cold chain**: items flagged `cold_chain` must ride in a refrigerated vehicle. A violation is a food-safety incident, not a scheduling inconvenience.
- **Allergens**: any item allergen that matches a client allergen with severity `severe` is a hard stop. Your pipeline should refuse to load the item, not just warn.
- **Two-person clients**: clients with `requires_two_person` true need a two-person team. A solo driver means the delivery cannot happen at the door.
- **Driver limits**: each driver has a max hours per week and max distance per week. Exceeding these produces burnout and legal issues. Your optimizer should treat them as hard constraints.
- **Volunteer-led operation**: the humans running this read the output on a phone, between stops, often tired. Cognitive-load-lite interfaces win over dashboard sprawl. One clear "today's anomalies" panel beats seven charts.

## Suggested tech stack

- **Data**: pandas, duckdb for ad-hoc SQL over the parquet files.
- **Optimization**: Google OR-Tools (`pywrapcp`) for the VRP, open-source and free.
- **UI**: streamlit plus pydeck for maps. Flask or Next.js if you want something more custom.
- **Portability**: consider exporting in Onfleet or Routific JSON shape so your output can plug into real delivery software the ops lead might already use.
- **ML**: scikit-learn, XGBoost, or Prophet for forecasting. Keep it boring and interpretable.

## Where to go from here

- `notebooks/00_quickstart.ipynb` loads the data, validates the schema, and runs a baseline route quality scorer.
- `docs/erd.mmd` has the Mermaid entity-relationship diagram.
- `dictionary/fields.csv` has every column, type, and plain-english description.

Start with the notebook. Pick your angle by the end of day one. Ship something clickable by demo time.
