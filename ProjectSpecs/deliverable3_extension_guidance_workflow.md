# Deliverable 3 Extension Guidance Workflow Setup

## Purpose
This note fixes the clean Deliverable 3 workflow for turning a comp-based salary anchor into provisional extension guidance without over-committing before Deliverable 1 is integrated.

The guiding separation is:

- **Valuation** = protected price / fair price / walk-away max
- **Action** = offer now / offer now but stay disciplined / wait and save flexibility / avoid overcommitting

This note is intentionally conservative. It treats the current output as a **provisional guidance engine based only on Block 2 comps and risk proxies**, not as the finished recommendation layer.

## Current inputs used
Primary file:
- `deliverable3_block2_archetype_comp_market_context.csv`

Supporting logic already embedded in that file:
- macro archetype
- shot-style subtype
- hybrid archetype label
- prototype fit ambiguity
- identity drift class
- realistic comp list
- comp salary match count / rate
- weighted mean comp salary anchor
- p25 / p50 / p75 comp salary anchors
- comp salary support band
- plain-language Block 2 market interpretation

## Step 1 — Build the market anchor band
For each player, define three internal price levels:

- **Protected price** = `comp_salary_anchor_p25`
- **Fair price** = `comp_salary_anchor_weighted_mean`  
  - fallback to `comp_salary_anchor_p50` if the weighted mean is missing
- **Walk-away max** = `comp_salary_anchor_p75`  
  - fallback to fair price if p75 is missing

Interpretation:
- Protected price = cautious opening number
- Fair price = internal estimate of reasonable value
- Walk-away max = hard cap beyond which the team should stop

These are **cap-percentage prices**, not raw dollars.

## Step 2 — Decision-card structure
The decision card should carry the following fields:

1. Player identity
   - PLAYER_NAME
   - macro archetype
   - shot-style subtype
   - hybrid archetype label

2. Market band
   - protected price
   - fair price
   - walk-away max
   - anchor band width

3. Evidence quality
   - comp salary match count
   - comp salary match rate
   - comp salary support band
   - final comp support bucket

4. Risk proxies from Block 2
   - prototype fit ambiguity
   - ambiguity band
   - identity drift class
   - risk proxy note

5. Strategic interpretation
   - scarcity tier
   - scarcity wording
   - provisional action bucket
   - provisional action reason

## Step 3 — Comp-support logic
Use a simple four-level evidence bucket:

- **strong**
  - support band = `high`
  - comp salary match count >= 4
  - comp salary match rate >= 0.8
  - prototype ambiguity not in the extreme tail
  - realistic comp neighborhood not unusually weak

- **moderate**
  - usable anchor exists
  - but support is not strong enough to justify an aggressive push

- **weak**
  - only one comp salary match or very weak support

- **insufficient**
  - no usable fair price from current Block 2 evidence
  - or no comp salary matches

This support bucket is meant to answer:
**“How much do we trust the comp-based market anchor?”**

## Step 4 — Scarcity / replaceability wording
The current conservative wording is:

- **High-Usage Primary Creators** → scarce
- **Scoring Bigs / Two-Way Forwards** → selective premium
- **Perimeter Wings & Connectors** → replaceable middle
- **Low-Usage Interior Bigs** → replaceable
- **Fringe / Low-Opportunity Players** → highly replaceable

This wording is only a strategic overlay. It is not allowed to override weak evidence.

## Step 5 — Provisional action buckets (Block 2 only)
Because Deliverable 1 is not yet integrated, these are **provisional** buckets.

### 1. `offer_now`
Use when:
- comp support is strong
- role family is scarce or selective premium
- role fit is relatively clean

Meaning:
- the team can act early because the current comp-based evidence is credible enough to justify moving before the market improves.

### 2. `offer_now_disciplined_band`
Use when:
- comp support is usable
- but not clean enough for an aggressive push

Meaning:
- offer now only inside the protected-to-fair band
- treat the walk-away max as a hard ceiling

### 3. `wait_and_save_flexibility`
Use when:
- comp support is moderate
- or the role read is still noisy
- or evidence is incomplete but the role is still strategically interesting

Meaning:
- do not take the player off the board
- but preserve timing value until forecast and risk outputs sharpen the case

### 4. `avoid_overcommitting`
Use when:
- evidence is weak or insufficient
- and/or the role is replaceable enough that the team should not pay ahead of support

Meaning:
- no aggressive early extension posture from current information alone

## Step 6 — What gets added later from Deliverable 1
This current workflow is deliberately incomplete.

Once Deliverable 1 is integrated, the final recommendation layer should adjust the fair value and the action bucket using:
- Years 5–7 forecast strength
- sleeper / neutral / bust outlook
- uncertainty band
- durability band

Then the final action rule becomes:

**Adjusted fair value = market anchor + upside premium - downside discount - durability penalty - uncertainty penalty**

After that, the final recommendation can be upgraded from provisional guidance to the finished extension stance.

## Files created in this step
- `deliverable3_provisional_extension_guidance_scaffold.csv`
- this note: `deliverable3_extension_guidance_workflow.md`

## Current cohort counts from provisional scaffold
- offer_now: 99
- offer_now_disciplined_band: 273
- wait_and_save_flexibility: 452
- avoid_overcommitting: 234

These counts are not final recommendations. They are the current Block 2-only starting posture.
