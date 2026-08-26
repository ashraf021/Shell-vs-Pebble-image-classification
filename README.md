# Shell vs Pebble Image Classification Using CNN

A binary image-classification project that identifies whether an input image contains a **Pebble** or a **Shell**. The submitted notebook uses **TensorFlow/Keras** with **MobileNetV2 transfer learning**, followed by a small custom classification head.

> **Notebook:** `shell-vs-pebble-image-classification-using-cnn.ipynb`

## 1. Project at a glance

| Item | Details |
|---|---|
| Task | Binary image classification |
| Classes | `Pebbles`, `Shells` |
| Framework | TensorFlow / Keras |
| Transfer-learning model | MobileNetV2, ImageNet weights |
| Input size | `224 × 224 × 3` RGB |
| Batch size | 32 |
| Training epochs | 10 |
| Optimizer | Adam |
| Learning rate | `1e-5` |
| Loss | Categorical cross-entropy |
| Test accuracy | **83.66%** |
| Test loss | **0.37694** |

## 2. Dataset

The notebook uses the Kaggle dataset arranged as:

```text
/kaggle/input/datasets/
└── vencerlanz09/
    └── shells-or-pebbles-an-image-classification-dataset/
        ├── Shells/
        └── Pebbles/
```

The dataset contains **4,284 images**:

- **2,743 Pebbles**
- **1,541 Shells**

The classes are read automatically from the parent directory names.

### Dataset split

The notebook first performs an **80/20 stratified train/test split**:

- Training pool: **3,427 images**
- Test set: **857 images**

The training pool is then split internally by `ImageDataGenerator(validation_split=0.20)`:

- Training: **2,742 images**
- Validation: **685 images**
- Test: **857 images**

The split uses `random_state=42` and stratification by class.

## 3. Preprocessing

The notebook applies the following preprocessing:

1. Finds image files with `.jpg`, `.jpeg`, and `.png` extensions.
2. Builds a Pandas DataFrame containing file paths and labels.
3. Resizes images to **224 × 224**.
4. Converts images to **RGB**.
5. Applies `tf.keras.applications.mobilenet_v2.preprocess_input`.
6. Uses categorical labels for the two classes.
7. Shuffles training and validation batches with seed `42`.
8. Does **not** apply additional image augmentation in the submitted notebook.

## 4. Model architecture

The project uses **MobileNetV2** as a pretrained feature extractor:

```text
Input: 224 × 224 × 3 RGB
        │
        ▼
MobileNetV2
ImageNet pretrained weights
include_top=False
pooling="avg"
        │
        ▼
Dense(128, ReLU)
        │
Dropout(0.20)
        │
        ▼
Dense(128, ReLU)
        │
Dropout(0.20)
        │
        ▼
Dense(2, Softmax)
        │
        ├── Pebbles
        └── Shells
```

The MobileNetV2 base is frozen:

```python
pretrained_model.trainable = False
```

The custom classification head is trained.

The notebook reports:

- **Total parameters:** 2,438,722
- **Trainable parameters:** 180,738
- **Non-trainable parameters:** 2,257,984

## 5. Training approach

The model is compiled with:

```python
Adam(learning_rate=1e-5)
```

and:

```python
loss="categorical_crossentropy"
metrics=["accuracy"]
```

Training is configured for **10 epochs** with:

- `EarlyStopping`
  - monitors `val_loss`
  - patience = 5
  - restores best weights
- `ModelCheckpoint`
  - monitors `val_accuracy`
  - saves only the best weights
- TensorBoard logging

The checkpoint filename used by the notebook is:

```text
shell_and_pebbles_classification_model.weights.h5
```

## 6. Results

The submitted notebook reports the following test performance:

```text
Test Loss: 0.37694
Test Accuracy: 83.66%
```

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Pebbles | 0.85 | 0.90 | 0.88 | 549 |
| Shells | 0.80 | 0.73 | 0.76 | 308 |
| **Accuracy** | | | **0.84** | **857** |
| Macro avg | 0.83 | 0.81 | 0.82 | 857 |
| Weighted avg | 0.83 | 0.84 | 0.83 | 857 |

The notebook's exact accuracy from the classification report is approximately **0.836639 (83.66%)**.

## 7. Output preview

The repository includes screenshots extracted from the executed notebook so that the project can be understood without opening the notebook first.

### Sample dataset images

![Sample dataset images](assets/sample_dataset_images.png)

### Training and validation loss

![Training and validation loss](assets/training_loss_curve.png)

### Training and validation accuracy

![Training and validation accuracy](assets/training_accuracy_curve.png)

### Confusion matrix

![Confusion matrix](assets/confusion_matrix.png)

### Sample predictions

The notebook displays random test images with both the true and predicted labels.

![Sample predictions](assets/sample_predictions.png)

### Grad-CAM examples

The notebook also contains an optional Grad-CAM visualization for interpreting which image regions contribute to the prediction.

![Grad-CAM examples](assets/gradcam_examples.png)

## 8. Example predictions

The first 10 predictions produced by the submitted notebook were:

```text
['Pebbles', 'Shells', 'Pebbles', 'Pebbles', 'Pebbles',
 'Pebbles', 'Shells', 'Shells', 'Pebbles', 'Shells']
```

## 9. Requirements

The notebook imports:

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

A convenient installation command is:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

For the exact executed notebook environment, the recorded TensorFlow version was:

```text
TensorFlow 2.20.0
```

## 10. How to run on Kaggle

The notebook is written as a **Kaggle-ready** project.

### Step 1 — Create/open a Kaggle notebook

Upload:

```text
shell-vs-pebble-image-classification-using-cnn.ipynb
```

### Step 2 — Add the dataset

In Kaggle, use **Add Data** and attach the Shells-or-Pebbles dataset so that the following directory is available:

```text
/kaggle/input/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset/
```

The notebook checks `/kaggle/input` and raises an error if `/kaggle/input/datasets` is missing.

### Step 3 — Run all cells

Run the notebook from top to bottom.

The notebook will:

1. Check the dataset directory.
2. Count the images.
3. Build the image DataFrame.
4. Display sample images.
5. Split the dataset.
6. Create training/validation/test generators.
7. Load pretrained MobileNetV2.
8. Build and compile the classifier.
9. Train the model.
10. Evaluate it on the test set.
11. Plot loss and accuracy curves.
12. Generate predictions.
13. Print a classification report.
14. Display a confusion matrix.
15. Display sample predictions.
16. Optionally generate Grad-CAM visualizations.

## 11. How to run locally

The dataset path in the submitted notebook is Kaggle-specific. For a local run, place the dataset in a directory with the same two class folders:

```text
dataset/
├── Shells/
└── Pebbles/
```

Then change:

```python
dataset = "/kaggle/input/datasets"
```

to your local dataset directory, for example:

```python
dataset = "./dataset"
```

The rest of the notebook can then be adapted to the local environment.

## 12. Repository structure

```text
shell-vs-pebble-cnn-repository/
├── README.md
├── shell-vs-pebble-image-classification-using-cnn.ipynb
└── assets/
    ├── sample_dataset_images.png
    ├── training_loss_curve.png
    ├── training_accuracy_curve.png
    ├── confusion_matrix.png
    ├── sample_predictions.png
    └── gradcam_examples.png
```

## 13. Notes and limitations

- This README documents the **submitted notebook as executed**; it does not claim results from a different training run.
- The dataset is imbalanced toward the `Pebbles` class (2,743 vs. 1,541 images).
- The submitted notebook freezes the MobileNetV2 feature extractor and trains only the custom classification head.
- No additional image augmentation is configured in the submitted notebook.
- The optional Grad-CAM section is included for model interpretability.

## 14. Project goal

The goal of this project is to demonstrate a complete binary image-classification workflow using transfer learning:

**Dataset → preprocessing → train/validation/test split → MobileNetV2 → training → evaluation → predictions → confusion matrix → Grad-CAM**

This makes the project suitable as a Deep Image Processing / CNN project demonstrating practical computer-vision classification.
