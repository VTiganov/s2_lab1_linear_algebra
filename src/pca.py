from linear_algebra import Matrix

from typing import List, Tuple
import math 


def center_data(X: Matrix) -> Matrix:
    '''DONE + TESTS'''
    means = [sum(X.get_column(j)) / X.rows for j in range(X.cols)]
    centered = [
        [X.data[i][j] - means[j] for j in range(X.cols)]
        for i in range(X.rows)
    ]
    return Matrix(centered)
    

def covariance_matrix(X_centered: Matrix) -> Matrix:
    '''DONE + TESTS'''
    n = X_centered.rows
    X_T = X_centered.transpose()
    product = X_T.multiply(X_centered)
    return product.multiply(1 / (n - 1))

def characteristic_polynomial(C: 'Matrix', lam: float) -> float:
    '''DONE + TESTS'''
    n = C.rows
    mat = Matrix([row.copy() for row in C.data])
    for i in range(n):
        mat.data[i][i] -= lam

    return mat.determinant()

def find_eigenvalues(C: Matrix, tol: float = 1e-8, max_iter: int = 100) -> List[float]:
    n = C.rows
    eigenvalues = []

    # Handle empty and non-square matrices
    if n == 0 or C.cols != n:
        return []

    # For triangular matrices, eigenvalues are diagonal elements
    if is_triangular(C):
        return sorted([C.data[i][i] for i in range(n)])

    # For block diagonal matrices, find eigenvalues of each block
    if is_block_diagonal(C):
        blocks = get_blocks(C)
        for block in blocks:
            eigenvalues.extend(find_eigenvalues(block, tol, max_iter))
        return sorted(eigenvalues)

    # General case: use bisection method
    def f(lam):
        return characteristic_polynomial(C, lam)

    # Estimate search range using Gershgorin circle theorem
    radius = sum(abs(C.data[i][j]) for i in range(n) for j in range(n) if i != j)
    start = min(C.data[i][i] - radius for i in range(n))
    end = max(C.data[i][i] + radius for i in range(n))

    print(f"Search range: ({start}, {end})")

    # Find intervals where sign changes occur or values are near zero
    intervals = find_sign_changes(f, start, end, n*20, tol)  # Increase the number of points
    print(f"Intervals: {intervals}")

    # Apply bisection to each interval
    for a, b in intervals:
        root = bisect(f, a, b, tol, max_iter)
        if root is not None:
            eigenvalues.append(root)

    # Check diagonal elements (important for multiple eigenvalues)
    for i in range(n):
        lam = C.data[i][i]
        if abs(f(lam)) < tol and not any(abs(lam - ev) < tol for ev in eigenvalues):
            eigenvalues.append(lam)

    # Remove duplicates and sort in ascending order
    unique_eigenvalues = []
    for ev in sorted(eigenvalues):
        if not unique_eigenvalues or abs(ev - unique_eigenvalues[-1]) >= tol:
            unique_eigenvalues.append(ev)

    print(f"Eigenvalues: {unique_eigenvalues}")
    return unique_eigenvalues

def is_triangular(C: Matrix) -> bool:
    """Check if matrix is upper triangular"""
    n = C.rows
    for i in range(1, n):
        for j in range(i):
            if abs(C.data[i][j]) > 1e-10:
                return False
    return True

def is_block_diagonal(C: Matrix) -> bool:
    """Check if matrix is block diagonal"""
    n = C.rows
    # Simple check for obvious block structure
    # More sophisticated detection could be added
    for i in range(n):
        for j in range(n):
            if i != j and abs(C.data[i][j]) > 1e-10:
                if not (abs(i-j) == 1 and abs(C.data[i][j] - 1) < 1e-10):
                    return False
    return True

def get_blocks(C: Matrix) -> List[Matrix]:
    """Extract blocks from block diagonal matrix"""
    blocks = []
    n = C.rows
    i = 0
    while i < n:
        j = i
        while j < n and (j == i or C.data[j][j-1] == 1):
            j += 1
        block = [row[i:j] for row in C.data[i:j]]
        blocks.append(Matrix(block))
        i = j
    return blocks

def find_sign_changes(f, start, end, num_points, tol=1e-8):
    intervals = []
    step = (end - start) / num_points
    x1 = start
    f1 = f(x1)

    for _ in range(num_points):
        x2 = x1 + step
        f2 = f(x2)

        if f1 * f2 <= 0 or abs(f1) < tol or abs(f2) < tol:  # Sign change or near-zero value
            intervals.append((x1, x2))

        x1, f1 = x2, f2

    return intervals

def bisect(f, a, b, tol, max_iter):
    """Bisection method for root finding"""
    fa, fb = f(a), f(b)
    
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    
    if fa * fb > 0:
        return None  # No root in interval
    
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        
        if abs(fc) < tol:
            return c
            
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
            
        if b - a < tol:
            return (a + b) / 2
    
    return (a + b) / 2

def find_eigenvectors(C: 'Matrix', eigenvalues: List[float]) -> List['Matrix']:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    #TODO
    pass

def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вход:
    eigenvalues: список собственных значений
    k: число компонент
    Выход: доля объяснённой дисперсии
    """
    #TODO
    pass

def pca(X: 'Matrix', k: int) -> Tuple['Matrix', float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """
    pass

def plot_pca_projection(X_proj: 'Matrix'):
    
    """
    Вход: проекция данных X_proj (n×2)
    Выход: объект Figure из Matplotlib
    """
    pass

def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    
    """
    Вход:
    X_orig: исходные данные (n×m)
    X_recon: восстановленные данные (n×m)
    Выход: среднеквадратическая ошибка MSE
    """
    pass

def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    
    """
    Вход:
    eigenvalues: список собственных значений
    threshold: порог объяснённой дисперсии
    Выход: оптимальное число главных компонент k
    """
    pass


def handle_missing_values(X: 'Matrix') -> 'Matrix':
    
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    pass


def add_noise_and_compare(X: 'Matrix', noise_level: float = 0.1):

    """
    Вход:
    X: матрица данных (n×m)
    noise_level: уровень шума (доля от стандартного отклонения)
    Выход: результаты PCA до и после добавления шума.
    В этом задании можете проявить творческие способности, поэтому выходные данные не
    типизированы.,→
    """
    pass


def apply_pca_to_dataset(dataset_name: str, k: int) -> Tuple['Matrix', float]:
    
    """
    Вход:
    dataset_name: название датасета
    k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    pass




