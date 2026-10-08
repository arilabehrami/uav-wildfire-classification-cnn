# Label policy and development evaluation, 2026-10-08

The task is to detect visible flames OR confirmed smoke from a fire. Fog,
clouds, dust, and scenes without fire are No_Fire. Unresolved smoke provenance
must not be automatically relabelled from a model score.

The user confirmed fire smoke in Test/No_Fire frames 1592-1913. Exactly 322
labels were changed in test_adapt_val_smoke_v1.csv; the original split remains
intact. The user subsequently confirmed that Test/No_Fire frames 0-1391 contain
fog/clouds without fire. Those 1392 training labels remain No_Fire. Confirmation
does not extend to other frame ranges, and test_final is not revisited.

This changes the meaning of earlier reported false positives on the confirmed
segment: they are detections of a target phenomenon under the updated task.
It does not demonstrate improved model weights. Smoke as a learned feature is
not itself evidence of shortcut learning when smoke is part of the target.
Confusion of smoke with fog/clouds remains a separate hypothesis to evaluate.

The corrected Test validation contains only Fire. A two-class development panel
therefore combines it with existing validation_temporal images. At most 200
images per legacy label/scene group (lake/forest) are selected evenly across the
available order, after removing identical SHA256 images and images with dHash
Hamming distance <=8 to the union of train_temporal, train_mixed_v3, and
train_with_adapt. Selection uses paths, labels, and hashes, never model scores.
The filter also covers the confirmed smoke images. No new images are moved
from training into validation. Hash distance is an approximate similarity
screen, not proof of scene independence.

These are existing development data already used in model selection, not a
new independent final test. Negative legacy labels are inherited rather than
newly confirmed by the user. Camera and class are partly confounded; combined
AUC cannot establish same-camera discrimination between smoke and fog.
Consecutive frames are correlated. Source-specific recall/specificity must be
reported alongside combined metrics.

v3 and v5 are evaluated with fresh inference: RGB, 254x254, scaled to [0,1],
one sigmoid score. The fixed threshold 0.5 is reported first. The 95% recall
threshold is exploratory calibration on this same development panel, not an
independent estimate of deployed performance. No retraining benefit may be
claimed from correcting validation labels alone.

A new recording containing confirmed fire smoke and confirmed fog/clouds
without fire is required for a credible final test of this distinction.

## Results on this development panel

The panel contains 1488 images: 322 confirmed smoke positives, 566 original
Test Fire positives, 200 legacy Fire positives, and 400 legacy No_Fire
negatives. Filtering excluded 240 candidate images near training before
the deterministic legacy subsampling step.

| Model | Mixed-domain AUC | Smoke recall at 0.5 | Overall Fire recall at 0.5 | No_Fire specificity at 0.5 |
| --- | --- | --- | --- | --- |
| v3 | 0.9575 | 70.19% | 87.41% | 91.25% |
| v5 | 0.9371 | 24.53% | 65.17% | 91.25% |

Exploratory calibration uses the highest threshold achieving at least 95%
empirical recall in each positive source group, then compares negative
specificity. For v3 the resulting threshold is 0.02003161795437336: smoke
306/322, original Test Fire 566/566, legacy Fire 197/200, and legacy No_Fire
358/400. Thus Fire recall is 98.25% overall and specificity is 89.5%, with
42 false alarms. At the same per-group recall target v5 gives specificity
66.5% (134 false alarms). v3 is selected for development use in the API.
These numbers describe the calibration data themselves, not an independent
performance estimate. The calibration rule was explored during this audit.

No model weights were retrained in this iteration. Existing v5 weights were
selected using the older label interpretation, and should not be treated as
evidence against all possible adaptation methods. A future training run uses
the corrected development split by default and preserves the training fog
labels, but model selection still needs new same-camera negatives to address
the smoke/fog question directly. The API uses a binary label; no validated
abstention/Uncertain operating region has been established.
