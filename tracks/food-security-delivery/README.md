# Track 2: Food Security Delivery Operations

Victoria has a growing population of mobility-limited residents who rely on delivered meals and grocery hampers. The operators running these routes are mostly volunteer-led. Route planning is manual. Dietary and allergy matching is tribal knowledge. Volunteer no-show and burnout rates are not systematically tracked. A missed delivery means a missed meal. An allergen-mismatched delivery could mean hospitalization.

## Quickstart

```bash
# From repo root
pip install -r requirements.txt
python tracks/food-security-delivery/generator/generate.py
jupyter lab tracks/food-security-delivery/notebooks/00_quickstart.ipynb
```

The generator produces:

- 9 Parquet files in `data/raw/`
- 1 SQLite database in `data/raw/track2.sqlite` with two pre-joined views: `v_route_performance` and `v_stop_details`
- 9 sample CSVs (first 1000 rows each) in `data/sample/`

## Data at a glance

| Table | Rows | What it is |
|---|---|---|
| depots | 2 | Central (downtown Victoria) + satellite (Saanich) |
| vehicles | 8 | Mix of car, van, cargo_van, bike. 1 wheelchair-lift (VEH-06), 5 refrigerated (includes VEH-01, a car with insulated-cooler kit) |
| drivers | 8 | 6 volunteer + 2 staff, with skill and availability flags |
| clients | 500 | Mobility-limited Victoria residents receiving delivered meals |
| delivery_requests | 10,000 | Standing + ad-hoc delivery requests with time windows, spread over the 3-week operating horizon |
| routes | 300 | Roughly 3 weeks of daily routes with planned vs actual. ~196 active, ~104 retained empty by design (no constraint-compatible request on that service_date) |
| route_stops | ~3,500 | Join table with actuals, failure reasons, signatures. Roughly 17 stops per active route, realistic MOW density |
| inventory_items | ~150 | Meal and hamper catalogue with allergen + dietary tags |
| delivery_request_items | ~24,000 | Line items per delivery |

See `dictionary/fields.csv` for every column.

## Domain anchors

- Meals on Wheels operations (hot + frozen programs)
- Grocery-hamper charity distribution
- Culturally specific meal programs
- Vehicle Routing Problem (Onfleet, Routific schemas)
- Cold-chain handling for refrigerated deliveries
- Volunteer-led operations (burnout, retention, no-show analytics)

Read `docs/problem-framing.md` for how the kit translates these into data.

## Golden join

`route_stops ⋈ delivery_requests ⋈ clients ⋈ routes ⋈ drivers` is the foundational join. The starter notebook builds it in cell 6. SQLite exposes it as `v_stop_details`. Start there.

## Seeded ground truth (for scoring your solutions)

- **3 allergen conflict violations** in `delivery_request_items`, tagged in request `notes` with `DATA_QUALITY_ISSUE: allergen conflict`. A good dietary-safety detector catches all three.
- **4 post-closure delivery attempts** tagged in `route_stops.driver_notes` with `DATA_QUALITY_ISSUE: delivered after closure`. A good status-gate catches these.
- **~5% of two-person-required clients** marked with failure `requires_two_person_unavailable`. A compatibility detector flags these as misassignments.

Your solution can self-score against these.

## Constraints your solution must respect

1. **Cold chain.** `delivery_request.cold_chain_required=True` means the assigned vehicle must have `refrigerated=True`. No exceptions.
2. **Severe allergens.** If a client has any `allergy_*_severity='severe'`, their line items cannot contain items with that allergen flag. This is a hard safety stop.
3. **Wheelchair clients.** If `client.mobility_wheelchair=True`, the assigned vehicle must have `wheelchair_lift=True` (only VEH-06 qualifies in the seed data).
4. **Closed or deceased clients.** `enrolment_status` in {closed, deceased} means no deliveries after `closure_date`. Solutions that schedule past closure are broken.
5. **Driver capabilities.** Drivers with `pet_allergy_flag=True` must not be assigned to clients with `has_dog_on_premises=True`. Drivers without matching language skills must not be assigned to clients where `interpreter_required=True`.
6. **Volunteer cognitive load.** This is a volunteer-led operation. Dashboards with 40 widgets are a fail regardless of technical merit. Build for the Tuesday morning volunteer lead.

Judges will check at least one of these.

## Baseline that ships

The starter notebook includes a greedy route quality scorer (weighted combination of on-time rate, completion rate, skill match, cold chain respect) and a nearest-neighbor savings heuristic that estimates route-level distance improvement. Your team should be able to beat this. Candidates: OR-Tools VRP, graph-based matching, reinforcement learning for dispatch policy.

## Extension ideas

- OR-Tools VRP optimizer with per-vehicle capacity + time-window constraints
- Driver-client compatibility matcher (skill + language + mobility)
- No-answer prediction per client (time-series features)
- Demand forecasting by `program_type` and season
- Volunteer scheduling with fairness constraints (rotating routes, hours parity)
- Low-cognitive-load dispatcher UI for a volunteer lead

See `docs/problem-framing.md` for the full treatment.
