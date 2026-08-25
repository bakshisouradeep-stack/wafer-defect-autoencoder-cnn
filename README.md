

## 📌 Project Overview

In semiconductor fabrication, spatial failure distributions on silicon substrates (wafer maps) encode highly specific information regarding root-cause process anomalies (e.g., plasma-etch imbalances, chemical vapor deposition variance, or mechanical scratches). 

This project establishes a robust computer vision pipeline utilizing **Double-Layer Convolutional Neural Networks (CNNs)** and **Autoencoders** to automatically segment, extract spatial signatures, and classify topologies from real-world fabrication data.

### Supported Topology Anomalies
The architecture is optimized to categorize the 8 industrial defect footprints found within the **WM811K dataset**:
*   **Center**: Failures clustered tightly at the center of the wafer.
*   **Donut**: Ring-shaped defect distributions with a clear core.
*   **Edge-Loc / Edge-Ring**: Symmetrical or localized marginal anomalies.
*   **Loc (Localized)**: High-density focal point defects.
*   **Scratch**: Linear directional signatures typical of tool/handling damage.
*   **Random**: Stochastic noise distributions across the die map.
*   **Near-full**: Severe macro-scale processing failures.

---

## 🔬 Framework & Architecture

The pipeline consists of three core engineering milestones:

1. **Data Preprocessing & Structural Normalization**
   * Dimensions within the WM811K dataset are highly variable (ranging across 632 unique shape permutations). Raw matrix data is dimensionally standardized to unified shapes (224 × 224 × 3) using bilinear area interpolation to retain pixel density configurations.
2. **Robustness Engineering via Data Augmentation**
   * Labeled wafer spaces are highly imbalanced, with native pattern defects comprising only 3.1% of the dataset (25,519 instances). To counteract severe overfitting, runtime augmentations are applied via the `imgaug` library, including bidirectional flipping, localized translations, spatial shearing, and uniform rotation steps across (-180°, 180°).
3. **Deep Convolutional Network Stack**
   * Implements a convolutional structure with Batch Normalization (`BatchNorm`) layers immediately preceding ReLU activation functions to stabilize inner covariate shifts.
   * Utilizes Spatial 2D Dropout layers to drop entire feature maps rather than individual pixels, preserving spatial structure while regularizing the internal layers.
   * **Bayesian Hyperparameter Optimization** is integrated to dynamically isolate performance boundaries across variable internal learning rates and kernel weights.

---

## 📊 Evaluation Metrics & Results

The network delivers high-performance evaluation benchmarks across highly imbalanced data matrices:

*   **Overall Classification Accuracy**: `98.0%`
*   **Overall Precision**: `93.0%`

### Confusion Matrix Insights
The model exhibits distinct structural precision when isolating macro-scale anomalies:
*   **Near-full** anomalies are classified with a high precision rate.
*   **Edge-Ring** signatures achieve exceptional topological partitioning due to symmetric spatial filters.

---

