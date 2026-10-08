# Visible-flame policy

Fire means visible flames, with or without smoke. No_Fire means no visible
flames, even when fire smoke is confirmed. Ambiguous flame/ember pixels are
Uncertain and excluded until reviewed. Review native images rather than just
thumbnails; do not infer flames from a scene being known to contain fire.

The 2026-10-08 smoke-positive correction and its threshold calibration are
historical work for a different target. They are not valid performance claims
for this policy. Original splits are retained. No blanket reversal of the 322
smoke images was applied without checking flame visibility.

review.html contains both classes from train_with_adapt, validation_temporal,
validation_extra_v3 and test_adapt_val. Initial decisions cover 24 evenly spaced
Fire examples and the user-confirmed frame585 example. No model weights have
been retrained. The production classifier remains the previous smoke model,
and the frontend displays this limitation.

The exporter in review.html saves reviewed_flame_labels.csv. Include a reason
for each decision. After complete review:

    python backend/training/prepare_reviewed_flame_splits.py --review PATH_TO_CSV

The importer refuses incomplete reviews. Filter near-duplicates across the
resulting train/validation lists before training; path disjointness alone does
not establish independence. test_final is neither read nor changed.

Default training now requires reviewed flame splits:

    python backend/training/train_adapted.py --dry-run
    python backend/training/train_adapted.py

The new run name is fire_flame_v6; existing model checkpoints are not overwritten.

A subsequent small pilot was trained. See PILOT.md for its data, results and limitations. The full audit remains incomplete; the pilot can be selected explicitly with FIRE_MODEL_VARIANT=flame_pilot.
