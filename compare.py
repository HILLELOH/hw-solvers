import numpy as np
from scipy.optimize import root
import time
import matplotlib.pyplot as plt

# Part A
def solve_with_root(a, b):
    """
    Solves the linear equation system Ax = b using scipy.optimize.root.

    Parameters:
    a (numpy.ndarray): A square matrix of coefficients (N x N).
    b (numpy.ndarray): A vector of dependent variables (length N).

    Returns:
    numpy.ndarray: The solution vector x.

    Examples:
    >>> import numpy as np
    >>> # Test 1: Simple 2x2 system
    >>> a1 = np.array([[3, 1], [1, 2]])
    >>> b1 = np.array([9, 8])
    >>> x1 = solve_with_root(a1, b1)
    >>> np.allclose(x1, [2., 3.])
    True
    
    >>> # Test 2: 3x3 Identity matrix (Edge case: Ax = b where A is I means x = b)
    >>> a2 = np.eye(3)
    >>> b2 = np.array([5.5, -2.0, 10.1])
    >>> x2 = solve_with_root(a2, b2)
    >>> np.allclose(x2, [5.5, -2.0, 10.1])
    True
    """
    return root(lambda x: np.dot(a, x) - b, np.zeros(len(b))).x

#Part B
def test_solve_with_root(num_tests=10):
    """
    Verifies the solve_with_root function against numpy.linalg.solve.
    Generates random N x N matrices and checks if both methods yield 
    the same solution (accounting for floating-point precision).
    """
    np.random.seed(42) # Ensures reproducibility
    
    for i in range(num_tests):
        n = np.random.randint(2, 20)
        a, b = np.random.rand(n, n), np.random.rand(n) # Generate both on one line
        
        # Calculate and assert in a single line
        assert np.allclose(solve_with_root(a, b), np.linalg.solve(a, b), atol=1e-6), f"Test {i+1} failed for matrix size {n}x{n}"
        
    print(f"All {num_tests} random verification tests passed successfully!\n")


#Part C
def compare_performance(sizes=[2, 10, 50, 100, 200, 300, 500, 750, 1000], trials=3):
    """
    Compares the average execution time of solve_with_root against np.linalg.solve 
    across different matrix sizes, plots a graph, and saves it as an image.
    """
    times_linalg, times_root = [], []
    
    print(f"Starting performance comparison (averaging over {trials} trials per size)...")
    for n in sizes:
        t_lin, t_rot = 0, 0
        
        # Run a few trials to get a true "average" execution time
        for _ in range(trials):
            a, b = np.random.rand(n, n), np.random.rand(n)
            
            t0 = time.time()
            np.linalg.solve(a, b)
            t_lin += time.time() - t0
            
            t0 = time.time()
            solve_with_root(a, b)
            t_rot += time.time() - t0
            
        times_linalg.append(t_lin / trials)
        times_root.append(t_rot / trials)
        print(f"  Size {n}x{n} completed.")

    # Create and save the graph using shorthand formatting
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_linalg, 'bo-', label='numpy.linalg.solve')
    plt.plot(sizes, times_root, 'rx-', label='scipy.optimize.root')
    
    plt.title('Performance Comparison: numpy.linalg.solve vs scipy.optimize.root')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison.png')
    print("\nGraph saved successfully as 'comparison.png'")

#main
if __name__ == '__main__':
    # 1. Run the doctest (Part A)
    import doctest
    print("Running Doctests...")
    doctest.testmod(verbose=False)
    print("Doctests completed.\n")
    
    # 2. Run the random tests (Part B)
    test_solve_with_root()
    
    # 3. Run performance comparison and generate graph (Part C)
    compare_performance()