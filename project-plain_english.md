# Alpha9alpha10 PAM Pipeline — Plain-English Overview

Identity-hiding explanation of what this project does, for a non-technical reader.

## What is this project about?

We are looking for a chemical "volume knob" for a type of receptor in the inner
ear and nervous system called the alpha9alpha10 nicotinic acetylcholine
receptor. Turning this knob up (with a drug candidate called a "PAM") could
treat chronic pain and hearing loss by making the receptor more sensitive to
its natural signal, without sedating the person.

The catch: the receptor looks structurally different from the classic
"nicotinic" receptors each new drug is usually measured against. Most
screening screens check against the wrong knob. We built our own.

## How we work (the 7-step process)

Computers and a chemist in the lab learn together, in a tight loop. No step
is skipped, and no step is guessed at.

1. **Learn from family history.** We started from 30 compounds related to
   vitamin C (ascorbate) that are known to touch this receptor channel. Seven
   of them turn the knob up. We studied what chemical features make the good
   ones good.

2. **Train a computer judge.** A machine-learning model reads those 30 examples
   and becomes a "judge" that scores any new chemical structure as more or less
   likely to be a good knob-turner. We freeze the judge when we are satisfied,
   and never change its mind later (that would be cheating on ourselves).

3. **Generate candidate chemicals.** Medicinal-chemistry scanning enumerates
   97 analog ideas from the 7 active knob-turners; 86 fail chemical
   validity checks (a fact we record, not hide). The survivors give 15 unique
   molecules (9 analogs + 6 negative controls). Then we check reality: are
   they easy to make? are they safe? do they look like drugs? Reality
   filters them down to 6 solid candidates.

4. **Cross-check for safety and selectivity.** We check the 6 against 70 other
   close-by receptor modulators to make sure our candidates will not grab the
   wrong target (which is what causes side effects). We also would like to run
   structural 3D models and route-to-synthesis suggestions; those tools are NOT
   installed, so we honestly say "not available" rather than guess or invent a
   route or a structure.

5. **Design a fair experiment.** We build two blinded plates: one arm gets our
   best computer-ranked candidates, one gets randomly picked comparison
   chemicals. Nobody sees the labels during the run. Numbers and success rules
   are written BEFORE the experiment, not after.

6. **Run the lab experiment.** A single lead chemist (with a doctor of
   medicinal chemistry) must sign off that each plate makes synthetic and
   scientific sense before anything is tested. Not yet done — the gate stands
   at "pending review."

7. **Learn from the lab and repeat.** Round results come back into the model,
   the hypotheses, and the next round's design. We repeat until we either find
   a strong, clean drug lead or the data says stop.

## Honest status (no fabrication, ever)

- All the computational steps (1–5) are complete and checked by automated
  tests: 19/19 pipeline blocks run clean.
- No experimental round has been run yet — every candidate is labeled
  "PENDING REVIEW" and the oversight gate blocks experiments until a chemist
  approves.
- Structural 3D models (AF3/Boltz) are "NOT AVAILABLE" because those tools
  aren't installed — we say so, and we never invent a structure that doesn't
  exist.
- The route-to-synthesis tool (AIZynthFinder) is NOT installed in this build.
  It has not suggested any routes (0 of 7 candidates have a plan), and we say
  so. A scaffolded phase is ready to produce route suggestions once the tool is
  installed — suggestions only, to help a chemist decide, never a promise that
  the chemistry will work exactly as predicted.
- Selectivity is reassuring so far: the frozen judge, when shown the 70
  close-relative chemicals, gives none of them a strong score (0 of 70).

## What the R01 grant wants to do with this

A single integrated Aim: medicinal chemists lead an analog program (five
planned synthesis studies) around the ascorbate pharmacophore, while an
AI-guided platform prioritizes and validates which planned candidates to
advance, in pre-registered, blinded laboratory rounds:

## Security / privacy notes

- All identity information about the PI lab and participants is deliberately
  excluded from these files.
- Seeds, hashes, and logs keep every claim traceable to its raw input.