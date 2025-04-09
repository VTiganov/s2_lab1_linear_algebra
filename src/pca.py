from linear_algebra import Matrix

from typing import List, Tuple
import math 

def center_data(X: Matrix) -> Matrix:
    means = [sum(X.get_column(j)) / X.rows for j in range(X.cols)]
    centered = [
        [X.data[i][j] - means[j] for j in range(X.cols)]
        for i in range(X.rows)
    ]
    return Matrix(centered)

def covariance_matrix(X_centered: Matrix) -> Matrix:
    n = X_centered.rows
    X_T = X_centered.transpose()
    product = X_T.multiply(X_centered)
    return product.multiply(1 / (n - 1))

def characteristic_polynomial(C: 'Matrix', lam: float) -> float:
    n = C.rows
    mat = Matrix([row.copy() for row in C.data])
    for i in range(n):
        mat.data[i][i] -= lam

    return mat.determinant()



def find_eigenvalues(C: Matrix, tol: float = 1e-6, max_iter: int = 100):
    n = C.rows
    eigenvalues = []

    if n == 0 or n != C.cols:
        return []

    def f(lam):
        return characteristic_polynomial(C, lam)

    return eigenvalues

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




