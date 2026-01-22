# Math Requirements — Precisely Scoped

You can divide the math into **three buckets**:

1. Math you must **understand deeply**
2. Math you must **recognize but not derive**
3. Math you can **ignore entirely**

---

## 1. Core Math You MUST Understand

### 1.1 Linear Algebra (Minimal, Practical)

You do **not** need abstract proofs.

You **do** need:

#### Vectors

* What a vector is
* Dot product (intuition, not proof)
* Vector as feature representation

Used in:

* Dense layers
* Logits before softmax

You should be able to understand:

```text
y = Wx + b
```

as:

> weighted combination of features

---

#### Matrices

* Matrix multiplication (shape logic)
* Why dimensions must align

Used in:

* Neural network layers
* Batch processing

If you understand **shapes**, you’re good.

---

### 1.2 Calculus (Optimization, Not Theory)

You do **not** need symbolic calculus mastery.

You **do** need:

#### Derivatives (intuition)

* What a derivative means
* Why gradients point in “direction of change”

Used in:

* Backpropagation
* Loss minimization

You should understand:

> “We change parameters to reduce error.”

---

#### Chain Rule (Conceptual)

* How changes propagate backward

Used in:

* Training neural networks

You do **not** need to derive backprop by hand.

---

### 1.3 Probability (Very Important)

This is **critical**.

#### Discrete probability

* What probabilities represent
* Sum of probabilities = 1

Used in:

* Softmax outputs

---

#### Conditional probability

Understand this sentence deeply:

> ( P(digit = k \mid image) )

This is **exactly** what your CNN outputs.

---

#### Confidence thresholds

* Why max probability matters
* Why low confidence must reject input

This controls system correctness.

---

### 1.4 Optimization Basics

You need to understand:

* Loss functions (cross-entropy)
* Gradient descent (idea)
* Learning rate (intuition)

Not required:

* Convexity proofs
* Advanced optimizers math

---

## 2. Math You Need to RECOGNIZE (Not Master)

### 2.1 Convolution Math (High-Level Only)

You should know:

* Convolution = sliding weighted sum
* Kernel extracts local patterns

You do **not** need:

* Fourier transforms
* Convolution theorem

---

### 2.2 Softmax Function

You should know:

* Converts logits → probabilities
* Emphasizes the largest value
* Sensitive to scale

You do **not** need:

* Jacobian derivation

---

### 2.3 Cross-Entropy Loss

You should understand:

* Penalizes confident wrong answers heavily
* Encourages correct high confidence

You do **not** need:

* Information theory background

---

## 3. Math for Vision & Segmentation (Non-ML)

This is **classical math**, not ML.

### 3.1 Geometry

* Bounding boxes
* Coordinates
* Area, width, height
* Aspect ratios

Used in:

* Digit segmentation
* Sorting left-to-right

---

### 3.2 Thresholding Logic

* Binary decisions
* Pixel intensity cutoffs

No probability theory here.

---

## 4. Math for Symbolic Parsing & Arithmetic

This is **discrete mathematics**, but light.

### 4.1 Formal Logic (Very Light)

* Tokens
* Rules
* Precedence

You don’t need automata theory yet.

---

### 4.2 Arithmetic Rules

* Operator precedence
* Associativity
* Integer arithmetic

This is exact math, not approximate.

---

## 5. Math You Can IGNORE for This Project

You do **not** need:

❌ Measure theory
❌ Bayesian inference
❌ Information theory
❌ Eigenvalues/eigenvectors
❌ Linear programming
❌ Convex optimization theory
❌ Probability distributions beyond categorical
❌ Transformers math
❌ RNN math

If someone tells you otherwise, they are overfitting advice.

---

## 6. One-Page Study Checklist (Actionable)

Study **only** these:

### Must understand

* Vectors, matrices (shape intuition)
* Gradient descent concept
* Conditional probability
* Softmax outputs
* Cross-entropy loss

### Must recognize

* Convolution idea
* Backprop concept
* Confidence thresholds

### Must apply

* Geometry for bounding boxes
* Deterministic arithmetic rules

---

## 7. Key Mental Model (Very Important)

> **ML math is about approximation.
> Symbolic math is about certainty.**
