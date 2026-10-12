# PCA on MNIST: Dimensionality Reduction with KNN

Part of my **#100DaysOfMachineLearning** challenge (Day 43).

This project applies **Principal Component Analysis (PCA)** to the MNIST handwritten digits dataset and compares KNN classification accuracy **before and after** dimensionality reduction. It also explores how accuracy changes with the number of principal components and visualizes the data in 2D and 3D.

---

## Objective

- Check whether PCA can shrink the feature space (784 → fewer) without losing much accuracy
- Find how many components are actually needed
- Understand why PCA is useful for **training**, **production**, and **visualization**

---

## Dataset

- **MNIST**: 60,000 grayscale images of handwritten digits (28×28 pixels)
- Source: [Kaggle: hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- Each image is flattened into **784 features** (one per pixel) plus a label (0-9)
- Split: **80% train (48,000)** / **20% test (12,000)**

---

## Techniques Used

| Step | Technique |
|------|-----------|
| Data loading | `idx2numpy` to read IDX files, `pandas` DataFrame |
| Preprocessing | Flattening images, `StandardScaler` |
| Dimensionality reduction | `PCA` (scikit-learn) |
| Model | `KNeighborsClassifier` (default k=5) |
| Evaluation | `accuracy_score` |
| Sweep | PCA with 1 → 783 components, accuracy tracked for each |
| Visualization | `matplotlib` (sample digit), `plotly` (2D and 3D PCA scatter) |

---

## Results

| Setup | Features | Accuracy |
|-------|----------|----------|
| KNN on raw pixels | 784 | **97.15%** |
| StandardScaler + PCA + KNN | 200 | **95.50%** |

Accuracy vs. number of components (from the sweep):

| Components | Accuracy |
|-----------|----------|
| 1 | ~26.3% |
| 10 | ~90.8% |
| 20 | ~95.0% |

**Takeaway:** about 20 components out of 784 already reach ~95% accuracy. Most pixels are redundant, and PCA captures the important variance in far fewer dimensions.

---

## Key Learnings

- PCA is great for **dimensionality reduction**, with a small accuracy trade-off for a much smaller feature space
- Fewer features mean **faster training and inference**, which helps in **production**
- It reduces redundancy and noise
- It lets us **visualize** high-dimensional data in 2D/3D
- Choosing the number of components is a balance between accuracy and efficiency

---

## Project Structure

```
.
├── Principle_Compenent_Analysis.ipynb   # Main notebook
└── README.md
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn plotly idx2numpy
```

### 3. Get the dataset

Download MNIST from [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset), or with the Kaggle CLI:

```bash
kaggle dataset download -d hojjatk/mnist-dataset
unzip -q mnist-dataset.zip -d mnist/
```

Update the file paths in the notebook if your folder differs:

```python
train_image_path = "/content/mnist/train-images.idx3-ubyte"
train_image_label_path = "/content/mnist/train-labels.idx1-ubyte"
```

### 4. Run the notebook

Open `Principle_Compenent_Analysis.ipynb` in Jupyter or Google Colab and run all cells.

---

## Tech Stack

Python · NumPy · Pandas · Scikit-learn · Matplotlib · Plotly

---

## Future Improvements

- Choose components by **explained variance ratio** (e.g. keep 95% variance)
- Compare accuracy **with and without** `StandardScaler`
- Measure training and prediction **time** before vs. after PCA
- Try other classifiers (SVM, Random Forest, Logistic Regression)
- Tune `k` in KNN with cross-validation

---

## Connect

Following my #100DaysOfMachineLearning journey? Let's connect on [LinkedIn](https://www.linkedin.com/in/<your-profile>).

⭐ If you found this useful, consider starring the repo!
