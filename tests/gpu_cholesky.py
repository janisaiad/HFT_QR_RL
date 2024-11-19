import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import cupy as cp

def benchmark_gpu_neural_network():
    """
    Benchmark GPU vs CPU performance using scikit-learn MLPClassifier
    Generates synthetic data and compares training time
    Returns execution times and plots comparison
    """
    # Generate synthetic dataset
    X, y = make_classification(n_samples=10000, n_features=20, 
                             n_informative=15, n_redundant=5,
                             random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # CPU Training
    mlp_cpu = MLPClassifier(hidden_layer_sizes=(100, 50),
                           max_iter=100,
                           random_state=42)
    
    start_cpu = time.time()
    mlp_cpu.fit(X_train, y_train)
    cpu_time = time.time() - start_cpu
    
    # GPU Training (using CuPy for data transfer)
    X_train_gpu = cp.asarray(X_train)
    y_train_gpu = cp.asarray(y_train)
    
    mlp_gpu = MLPClassifier(hidden_layer_sizes=(100, 50),
                           max_iter=100,
                           random_state=42)
    
    start_gpu = time.time()
    mlp_gpu.fit(cp.asnumpy(X_train_gpu), cp.asnumpy(y_train_gpu))
    gpu_time = time.time() - start_gpu
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(['CPU', 'GPU'], [cpu_time, gpu_time])
    plt.title('Neural Network Training Time: CPU vs GPU')
    plt.ylabel('Time (seconds)')
    plt.savefig('benchmark_results.png')
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'cpu_score': mlp_cpu.score(X_test, y_test),
        'gpu_score': mlp_gpu.score(X_test, y_test)
    }

# Run benchmark
results = benchmark_gpu_neural_network()
print(f"CPU Training Time: {results['cpu_time']:.2f} seconds")
print(f"GPU Training Time: {results['gpu_time']:.2f} seconds")
print(f"CPU Test Accuracy: {results['cpu_score']:.4f}")
print(f"GPU Test Accuracy: {results['gpu_score']:.4f}")


