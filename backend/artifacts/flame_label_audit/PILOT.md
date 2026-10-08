# Visible-flame pilot

96 candidates were selected evenly within existing train_with_adapt and
validation_temporal classes, excluding user-Uncertain paths. Native-size
montages were visually reviewed. 12 ambiguous flame/ember or colored-object
cases were excluded. The user-confirmed No_Fire frame585 was added to training.
Decisions are recorded in pilot_visual_review.csv, including inherited and
reviewed labels and reasons. Final lists have 40 training images (18 Fire,
22 No_Fire) and 45 validation images (24 Fire, 21 No_Fire).

No exact SHA256 duplicates or dHash matches at Hamming distance <=8 were found
between these two pilot lists, using fingerprints previously computed in this
session. This does not establish recording independence. Existing v3 used the
broader development recordings, and all models are assessed on a small,
correlated development sample. No test_final data were accessed.

MobileNetV2 ImageNet features were frozen; a new average/max-pooling sigmoid
head was trained using balanced source/class weights and two augmented views
per image. v1 ran 12 epochs. Its best epoch was the last, so v2 allowed 40;
early stopping ended at epoch 30 and selected epoch 22 at threshold 0.5.

| Model | Fire correct | No_Fire correct | Frame585 diagnostic |
| --- | --- | --- | --- |
| Existing v3 at app threshold | 24/24 | 20/21 | Fire |
| Pilot v1 | 21/24 | 15/21 | Fire |
| Pilot v2 | 21/24 | 20/21 | No_Fire |

Frame585 was used in pilot training, so its changed result is not an
independent success claim. v2 misses more Fire than the existing app on this
panel. No automatic replacement of the app default was made. This experiment
shows a working training path, not established flame-vs-smoke generalization.
The panel lacks a substantial set of smoke-only negatives from independent
recordings; obtaining or reviewing those remains necessary.

To test the experimental model locally, run from the project root in PowerShell:

    $env:FIRE_MODEL_VARIANT='flame_pilot'
    python -m uvicorn backend.app.main:app --reload --port 8000

Start the frontend in another terminal with cd frontend and npm run dev.
To use the previous default again, set FIRE_MODEL_VARIANT to smoke_v3 before
starting the backend. The experiment uses threshold 0.5 and reports version
fire_flame_pilot_v2_experimental.
