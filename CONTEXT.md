### 1. The Algorithms & Models
You are reproducing a comparison between two specific strategies:

* **The Challenger: GP-ARD (Gaussian Process with ARD)**
    * **Kernel:** Matérn 5/2 kernel ($\nu = 2.5$).
    * **ARD Mechanism:** You must ensure the kernel learns separate lengthscales ($\ell_i$) for each feature dimension to handle anisotropy (e.g., thickness being more important than rotation).
    * **Acquisition Function:** Lower Confidence Bound (LCB).
    * **Hyperparameter Optimization:** Type-II MLE using L-BFGS-B.
* **The Baseline: Random Selection**
    * A "dumb" strategy that selects the next experiment from the pool uniformly at random to serve as the control group.

### 2. The Datasets
Your slides commit to using the following datasets:
* **Primary:** **Crossed Barrel** (3D-printed polymer structures).
    * **Input:** 600 samples, 4 continuous dimensions.
    * **Target:** Mechanical toughness.
* **Secondary:** **AgNP** (Silver nanoparticle synthesis).
    * **Input:** 164 samples, 5 dimensions.
    * **Target:** Absorbance spectrum score.

### 3. The Experimental Protocol (The "Loop")
You must implement a **Pool-Based Active Learning** loop. The `framework_gp.py` file contains the skeleton for this, but you must ensure your reproduction follows these specific steps from your slides:

1.  **Initialization:** Start with $N_{initial} = 2$ randomly selected samples.
2.  **Iteration:**
    * Train the GP surrogate on the *Observed Set*.
    * Predict mean ($\mu$) and uncertainty ($\sigma$) for the *Unobserved Pool*.
    * Select the next candidate $x^*$ using LCB.
    * "Reveal" the label (move from Unobserved to Observed).
3.  **Repetition:** Perform this entire loop for **50 independent random seeds** to ensure statistical significance.

### 4. The Critical Discrepancy (Attention Required)
There is a conflict between your slides and the provided code regarding the LCB parameter ($\lambda$):
* **Your Slides:** State you will use **$\lambda = 1$** to balance exploration and exploitation.
* **The Code (`framework_gp.py`):** Uses `ac_value = LCB(X_j, GP_learn, 2)` (implying **$\lambda = 2$**).

**Recommendation:** Stick to your slide definition ($\lambda = 1$) as you cited it is "proven optimal in paper", or run both to see if the code's default ($\lambda = 2$) yields better results.

### 5. The Evaluation Metrics
You are **not** using standard ML metrics like Accuracy or Confusion Matrices (which are in the generic `performance_visualization.py` file). You must implement the specific discovery metrics defined in your slides:

* **Top-k% Coverage:** The ratio of elite materials found. Target is reaching 80% coverage of the top 5% of materials.
* **Enhancement Factor (EF):** The speedup of BO over Random at iteration $i$.
* **Acceleration Factor (AF):** The ratio of iterations required by Random vs. BO to reach the 80% coverage target.

### 6. The Visualizations
To "reproduce the results," you need to generate these specific plots:
1.  **Discovery Rate Plot:** Iteration number ($x$-axis) vs. Top 5% Coverage ($y$-axis), showing the median and IQR (shaded region) for both GP-ARD and Random.
2.  **Manifold Visualization:** A PCA projection of the dataset (using `manifold_visualization.py` as a base) to show the distribution of high-performing materials.

---

### Next Step
The provided code is a bit scattered (`framework_gp.py` has the logic, but `performance_visualization.py` has the wrong metrics).

**Would you like me to generate a clean, single-file Python script that consolidates `framework_gp.py` but updates the Acquisition Function to $\lambda=1$ and implements the correct "Top-k% Coverage" calculation?**