# Submission Checklist

**Demo Night:** Saturday April 25, 2026, 6:30 to 9:30 PM at UVic Bob Wright B150 Flury Hall.

Submit via Devpost. Presentations start 7:00 PM.

---

## Required

- [ ] **Public GitHub repo** with your code. Link posted on Devpost.
- [ ] **README** in your repo explaining: what you built, which track, what problem you solved, how someone else could run it.
- [ ] **Working demo.** Live preferred, recorded acceptable. 3 minutes max on stage.
- [ ] **Track declaration.** Pick one: Inter-Org Referral & Care Coordination, or Food Security Delivery Operations.
- [ ] **Privacy & safety statement.** One paragraph on how your solution respects the constraints for your track (consent, OCAP, allergen safety, dietary restrictions, volunteer cognitive load, etc.).
- [ ] **Team list** with 1-5 members on Devpost.

## Recommended

- [ ] A hosted demo URL (Streamlit Cloud, Vercel, Replit all free)
- [ ] A short screen recording (Loom, OBS) in case live fails
- [ ] A deck or two slides max showing the problem and your solution
- [ ] Evidence your baseline beat the kit's baseline (e.g., precision / recall numbers, route optimization delta)
- [ ] Attribution to the starter kit: "Data: BuildersVault Social Services Hackathon starter kit (synthetic), CC BY 4.0."

## Judging themes

Each theme is weighted roughly equally.

### Problem Fit
Does this solve a real operator pain? Could a Track 1 case manager or a Track 2 route lead actually use this on Monday morning? Does it match a frustration a real operator talked about at kickoff?

### Technical Merit
Is the approach sound? Does it scale past the sample? Are choices justified? Bonus for open-source libraries (Splink, OR-Tools, etc.) over closed vendor lock-in.

### Social Services Domain Grounding
Does the solution respect consent? Does it honor OCAP for Indigenous clients? Does it handle allergen severity correctly? Does it account for volunteer-led operations vs full-time staff? Did you read the problem-framing doc?

### Production Readiness
Could a frontline worker run this without a developer in the room? Is the interface low-cognitive-load? Are failure modes sensible (fails safe, not silent)? Is setup documented?

---

## What not to do

- Don't use real PII or scraped data. Stick to the starter kit's synthetic data or partner with an org for their real data (that is a separate conversation).
- Don't ship a solution that violates the track's constraints (OCAP override, cold chain breach, allergen match on severe). If your demo shows a wheelchair client being assigned to a vehicle without a lift, that's a miss, not a feature.
- Don't scope creep. A tight, well-executed solution beats a sprawling demo every time.
- Don't forget the privacy statement. Judges will ask for it if it's missing.

---

## Prizes

See Devpost for the prize categories and amounts. Each track has its own top prize. Cross-track awards for production readiness and social services domain grounding.

Questions: ping the event host Lautaro Cepeda or ask in the event Discord.
