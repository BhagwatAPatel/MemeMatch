# MemeMatch

Webcam app that reads your facial expression and matches it to a meme.

## Language

**Feature vector**:
The 52 MediaPipe blendshape scores for one face, in `FEATURE_NAMES` order, clipped to 0-1. Produced by `app.features.extract_features`; the model's input.

**Sample**:
One labelled **feature vector**: an expression label plus its 52 scores, tagged with the **session** it was recorded in. One row in the sample CSV.
_Avoid_: row (when speaking of the domain), example, record

**Session**:
One continuous recording run in a single setting (lighting, distance, head angle). Every **sample** belongs to exactly one session. Sessions, not individual samples, are what get divided between train, validation and test, so a held-out score reflects an unseen recording rather than near-duplicates of training frames.

**Sample store**:
The module (`training.samples.SampleStore`) that owns the sample CSV: appending, counting, loading, clearing. The only code that knows the file format.
_Avoid_: dataset (that's the train/validation/test tensors built *from* the store)

**Dataset**:
Train, validation and test tensors plus the label vocabulary, built from the **sample store** by `training.dataset.load_dataset`. The validation split chooses the best training epoch; the test split is touched only for the final score.

**Label**:
The expression class name of a **sample** (`neutral`, `happy`, `surprised`, `angry`, `sad`). Class ids are derived from the sorted label names at load time.

**Expression model**:
The trained network together with its **label** vocabulary, the feature contract it was trained against and the **sessions** it held out for validation and test, saved as one file. Turns a **feature vector** into a label plus a confidence. Avoid: MLP, checkpoint; and "classifier" for the model itself (see **Expression classifier**).

**Expression classifier**:
The live-inference wrapper (`app.expression_classifier.ExpressionClassifier`) that loads an **expression model** and turns each **feature vector** into a **prediction**. The model is the trained artifact; the classifier is what the app calls.

**Prediction**:
The **expression model**'s **label** and **confidence** for one frame's **feature vector**. Per-frame and jittery; nothing acts on it directly.

**Smoother**:
The module that turns a stream of **predictions** (or their absence, when no face is found) into a **confirmed expression**.

**Confirmed expression**:
The **label** the app currently acts on, or none. A label becomes confirmed once it wins a strict majority of the recent **predictions** with sufficient mean **confidence**; any change from one confirmed state to another (a different label, or none once the vote fails) also requires the previous one to have been held for a minimum time. Losing the face clears it immediately.
_Avoid_: current expression, detected expression (ambiguous between raw and confirmed)

**Confidence**:
The **expression model**'s softmax probability for its top **label** on one **feature vector**. Uncalibrated: it measures how peaked the model's output is, not how often it is right. The live threshold is chosen by comparing confidence on correct vs. wrong held-out predictions.
