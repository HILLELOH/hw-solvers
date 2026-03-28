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
    >>> a = np.array([[3, 1], [1, 2]])
    >>> b = np.array([9, 8])
    >>> x = solve_with_root(a, b)
    >>> np.allclose(x, [2., 3.])
    True
    """
    # Define the function f(x) = Ax - b (we are looking for the root f(x) = 0)
    def fun(x):
        return np.dot(a, x) - b
    
    # Initial guess - a vector of zeros of the size of b
    x0 = np.zeros(len(b))
    
    # Find the root
    sol = root(fun, x0)
    return sol.x

#Part B
def test_solve_with_root():
    """
    Tests the solve_with_root function on several random inputs and compares 
    the result to that of numpy.linalg.solve.
    """
    np.random.seed(42) # Set seed for consistent testing
    print("Starting random verification tests...")
    
    for _ in range(10):
        n = np.random.randint(2, 20)
        # Create a random matrix and a random vector
        a = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Solve using both methods
        x_root = solve_with_root(a, b)
        x_linalg = np.linalg.solve(a, b)
        
        # Ensure the results are identical (up to reasonable numerical precision)
        assert np.allclose(x_root, x_linalg, atol=1e-6), f"Test failed for size {n}"
        
    print("All random tests passed successfully!\n")

#Part C
def compare_performance():
    """
    Compares the execution time of solve_with_root against np.linalg.solve 
    across different sizes (1 to 1000), plots a graph, and saves it as an image.
    """
    # Sizes to check (in jumps to avoid excessively long runtimes)
    sizes = [2, 10, 50, 100, 200, 300, 500, 750, 1000]

    # random_sizes = np.random.randint(2, 1000, size=10)
    # # 2. Sort the sizes from smallest to largest so the line graph draws correctly!
    # sizes = np.sort(random_sizes)
    
    times_root = []
    times_linalg = []
    
    print("Starting performance comparison (this may take a few seconds)...")
    for n in sizes:
        a = np.random.rand(n, n)
        b = np.random.rand(n)
        
        # Measure time for numpy.linalg.solve
        start_time = time.time()
        np.linalg.solve(a, b)
        times_linalg.append(time.time() - start_time)
        
        # Measure time for scipy.optimize.root
        start_time = time.time()
        solve_with_root(a, b)
        times_root.append(time.time() - start_time)
        
        print(f"  Size {n}x{n} completed.")

    # Create the graph
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_linalg, label='numpy.linalg.solve', marker='o', color='blue')
    plt.plot(sizes, times_root, label='scipy.optimize.root', marker='x', color='red')
    
    plt.title('Performance Comparison: numpy.linalg.solve vs scipy.optimize.root')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Average Execution Time (Seconds)')
    plt.legend()
    plt.grid(True)
    
    # Save the graph
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