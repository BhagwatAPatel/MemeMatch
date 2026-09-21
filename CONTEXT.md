# MemeMatch

Webcam app that reads your facial expression and matches it to a meme.

## Language

**Feature vector**:
The 52 MediaPipe blendshape scores for one face, in `FEATURE_NAMES` order, clipped to 0-1. Produced by `app.features.extract_features`; the model's input.

**Sample**:
One labelled **feature vector**: an expression label plus its 52 scores. One row in the sample CSV.
_Avoid_: row (when speaking of the domain), example, record

**Sample store**:
The module (`training.samples.SampleStore`) that owns the sample CSV: appending, counting, loading, clearing. The only code that knows the file format.
_Avoid_: dataset (that's the train/test tensors built *from* the store)

**Dataset**:
Train/test tensors plus the label vocabulary, built from the **sample store** by `training.dataset.load_dataset`.

**Label**:
The expression class name of a **sample** (`neutral`, `happy`, `surprised`, `angry`, `sad`). Class ids are derived from the sorted label names at load time.
