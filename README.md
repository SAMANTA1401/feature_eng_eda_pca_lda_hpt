Let's dive into the mathematical concepts behind Principal Component Analysis (PCA).
Mathematical Formulation
Given a dataset X with n samples and d features, represented as a matrix:
X = [x1, x2, ..., xn] ∈ ℝⁿˣᵈ
where xi is a d-dimensional vector.
Goal
Find an orthogonal transformation matrix W that projects X onto a lower-dimensional space, retaining most of the information.
Step 1: Standardization
Standardize X to have zero mean and unit variance:
Z = (X - μ) / σ
where μ and σ are the mean and standard deviation of each feature.
Step 2: Covariance Matrix
Compute the covariance matrix Σ of Z:
Σ = (1/n) _ ZᵀZ
where Zᵀ is the transpose of Z.
Step 3: Eigenvalue Decomposition
Compute the eigenvalue decomposition of Σ:
Σ = UΛUᵀ
where:
U is an orthogonal matrix (UᵀU = I) containing eigenvectors.
Λ is a diagonal matrix containing eigenvalues.
Step 4: Sorting and Selection
Sort eigenvalues in descending order and select the top k eigenvectors:
W = U[:, 1:k]
Step 5: Transformation
Project X onto the selected eigenvectors:
X_pca = Z _ W
Mathematical Properties
Orthogonality: WᵀW = I (orthogonal matrix)
Eigenvalues: Λ represents the amount of variance explained by each component.
Eigenvectors: U represents the directions of the principal components.
Key Equations
Covariance matrix: Σ = (1/n) _ ZᵀZ
Eigenvalue decomposition: Σ = UΛUᵀ
Transformation: X_pca = Z _ W
heoretical Justifications
Maximizing variance: PCA maximizes the variance of the projected data.
Minimizing reconstruction error: PCA minimizes the reconstruction error of the original data.
/////////////////////////////////
Linear Discriminant Analysis (LDA) is a dimensionality reduction technique that's closely related to Principal Component Analysis (PCA). Here's a deep dive into the mathematical concepts behind LDA:
Mathematical Formulation
Given a dataset X with n samples, d features, and c classes:
X = [x1, x2, ..., xn] ∈ ℝⁿˣᵈ
y = [y1, y2, ..., yn] ∈ {1, 2, ..., c}
Goal
Find a linear transformation matrix W that projects X onto a lower-dimensional space, maximizing class separation.
Step 1: Compute Mean Vectors
Compute the mean vector for each class:
μ*k = (1/n_k) * ∑[x_i | y_i = k]
where n*k is the number of samples in class k.
Step 2: Compute Within-Class Scatter Matrix
Compute the within-class scatter matrix:
S_W = ∑[n_k * (μ*k - μ) * (μ*k - μ)ᵀ]
where μ is the overall mean vector.
Step 3: Compute Between-Class Scatter Matrix
Compute the between-class scatter matrix:
S_B = ∑[n_k * (μ*k - μ) * (μ*k - μ)ᵀ]
Step 4: Compute Eigenvalue Decomposition
Compute the eigenvalue decomposition of the matrix:
(S_W⁻¹ * S*B) = UΛUᵀ
where:
U is an orthogonal matrix containing eigenvectors.
Λ is a diagonal matrix containing eigenvalues.
Step 5: Sorting and Selection
Sort eigenvalues in descending order and select the top k eigenvectors:
W = U[:, 1:k]
Step 6: Transformation
Project X onto the selected eigenvectors:
X_lda = X * W
Mathematical Properties
Optimization criterion: LDA maximizes the ratio of between-class scatter to within-class scatter.
Eigenvalues: Λ represents the amount of class separation explained by each component.
Eigenvectors: U represents the directions of the discriminant axes.
Key Equations
Within-class scatter: S*W = ∑[n_k * (μ*k - μ) * (μ*k - μ)ᵀ]
Between-class scatter: S_B = ∑[n_k * (μ*k - μ) * (μ*k - μ)ᵀ]
Eigenvalue decomposition: (S_W⁻¹ * S_B)
= UΛUᵀ
Theoretical Justifications
Bayes' theorem: LDA is related to Bayes' theorem, which provides a probabilistic framework for classification.
Gaussian assumption: LDA assumes Gaussian distributions for each class.
