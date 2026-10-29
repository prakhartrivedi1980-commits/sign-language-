
"""
Phase 1 - Neural Network Fundamentals
Module: Tensor Demonstrations

Project: Sign Language Recognition and Translation Assistant

Demonstrates:
1. Scalar, vector, matrix, and 3D tensor
2. Tensor shapes and dimensions
3. Matrix multiplication
4. Broadcasting
5. Reduction operations
6. Incorrect broadcasting
7. Neural-network layer computation
"""

import numpy as np


def demonstrate_tensors():
    print("\n" + "=" * 55)
    print("SIGN LANGUAGE ASSISTANT - TENSOR FUNDAMENTALS")
    print("=" * 55)

    # --------------------------------------------------
    # 1. SCALAR (0D TENSOR)
    # --------------------------------------------------
    scalar = np.array(10)

    print("\n1. SCALAR")
    print("Value:", scalar)
    print("Shape:", scalar.shape)
    print("Dimensions:", scalar.ndim)

    # --------------------------------------------------
    # 2. VECTOR (1D TENSOR)
    # --------------------------------------------------
    vector = np.array([1, 2, 3, 4])

    print("\n2. VECTOR")
    print("Value:", vector)
    print("Shape:", vector.shape)
    print("Dimensions:", vector.ndim)

    # --------------------------------------------------
    # 3. MATRIX (2D TENSOR)
    # --------------------------------------------------
    matrix = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("\n3. MATRIX")
    print(matrix)
    print("Shape:", matrix.shape)
    print("Dimensions:", matrix.ndim)

    # --------------------------------------------------
    # 4. THREE-DIMENSIONAL TENSOR
    # --------------------------------------------------
    tensor_3d = np.arange(24).reshape(2, 3, 4)

    print("\n4. THREE-DIMENSIONAL TENSOR")
    print(tensor_3d)
    print("Shape:", tensor_3d.shape)
    print("Dimensions:", tensor_3d.ndim)

    # --------------------------------------------------
    # 5. MATRIX MULTIPLICATION
    # --------------------------------------------------
    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [5, 6],
        [7, 8]
    ])

    result = A @ B

    print("\n5. MATRIX MULTIPLICATION")
    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Result:\n", result)
    print("Result shape:", result.shape)

    # --------------------------------------------------
    # 6. BROADCASTING
    # --------------------------------------------------
    weights = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    bias = np.array([10, 20, 30])

    broadcast_result = weights + bias

    print("\n6. BROADCASTING")
    print("Weights shape:", weights.shape)
    print("Bias shape:", bias.shape)
    print("Result:\n", broadcast_result)

    # --------------------------------------------------
    # 7. REDUCTION OPERATIONS
    # --------------------------------------------------
    data = np.array([
        [1, 5, 3],
        [4, 2, 6]
    ])

    print("\n7. REDUCTION OPERATIONS")
    print("Sum:", np.sum(data))
    print("Mean:", np.mean(data))
    print("Maximum:", np.max(data))
    print("Argmax:", np.argmax(data))
    print("Row sums:", np.sum(data, axis=1))
    print("Column means:", np.mean(data, axis=0))

    # --------------------------------------------------
    # 8. INCORRECT BROADCASTING
    # --------------------------------------------------
    print("\n8. INCORRECT BROADCASTING")

    wrong_bias = np.array([10, 20])

    try:
        invalid_result = weights + wrong_bias
        print(invalid_result)
    except ValueError as error:
        print("Expected broadcasting error caught!")
        print("Reason:", error)

    # --------------------------------------------------
    # 9. NEURAL NETWORK LAYER EXAMPLE
    # --------------------------------------------------
    print("\n9. NEURAL NETWORK LAYER EXAMPLE")

    # 2 samples, 3 input features each
    X = np.array([
        [0.5, 0.2, 0.1],
        [0.9, 0.4, 0.7]
    ])

    # 3 input features, 2 output neurons
    W = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ])

    b = np.array([0.1, 0.2])

    Z = X @ W + b

    # ReLU activation
    output = np.maximum(0, Z)

    print("Input shape:", X.shape)
    print("Weights shape:", W.shape)
    print("Bias shape:", b.shape)
    print("Output shape:", output.shape)
    print("Output:\n", output)

    print("\n" + "=" * 55)
    print("TENSOR DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 55)


if __name__ == "__main__":
    demonstrate_tensors()
