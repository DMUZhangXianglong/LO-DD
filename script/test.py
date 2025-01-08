# Simulating and calculating results for the modified scenario
# Importing the required libraries again in case of re-execution
import numpy as np
import matplotlib.pyplot as plt

# Simulating tunnel point cloud normals with slight z-direction constraint
num_points = 1000
n_vectors = np.zeros((num_points, 3))  # Initialize normals
n_vectors[:, 1] = 1  # Majority of normals pointing along the y-axis
n_vectors[:, 2] = np.random.uniform(-0.1, 0.1, num_points)  # Small variation in z-direction

# Normalize the vectors to ensure unit length
n_vectors = n_vectors / np.linalg.norm(n_vectors, axis=1, keepdims=True)

# Construct the Hessian matrix Htt
Htt = np.zeros((3, 3))
for n in n_vectors:
    Htt += np.outer(n, n)

# Perform eigen decomposition of Htt
eigvals, eigvecs = np.linalg.eigh(Htt)

# Project normals onto eigenvectors
projections = np.dot(n_vectors, eigvecs)

# Analyze projections
mean_projections = np.mean(projections, axis=0)
std_projections = np.std(projections, axis=0)

# Plot results
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.hist(projections[:, i], bins=30, alpha=0.6, label=f'Projection on eigenvector {i+1} (λ={eigvals[i]:.2f})')
plt.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero line')
plt.xlabel('Projection Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Projections on Htt Eigenvectors (with z-constraint)')
plt.legend()
plt.grid()
plt.show()

# Print analysis results
Htt, eigvals, eigvecs, mean_projections, std_projections
print(eigvals)
print(eigvecs)