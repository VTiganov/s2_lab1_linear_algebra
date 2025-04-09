import pytest
from linear_algebra import Matrix
from pca import center_data, covariance_matrix, characteristic_polynomial, find_eigenvalues 

def test_center_data():
    X = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    X_centered = center_data(X)
    expected = [
        [-2, -2],
        [0, 0],
        [2, 2]
    ]
    for row_c, row_e in zip(X_centered.data, expected):
        for v_c, v_e in zip(row_c, row_e):
            assert abs(v_c - v_e) < 1e-6

def test_covariance_matrix():
    X_centered = Matrix([
        [-2, -2],
        [0, 0],
        [2, 2]
    ])
    cov = covariance_matrix(X_centered)
    expected = [
        [4.0, 4.0],
        [4.0, 4.0]
    ]
    for row_c, row_e in zip(cov.data, expected):
        for v_c, v_e in zip(row_c, row_e):
            assert abs(v_c - v_e) < 1e-6


def test_characteristic_polynomial_diagonal():
    # Test with a diagonal matrix (eigenvalues are 1, 2, 3)
    C = Matrix([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3]
    ])
    assert characteristic_polynomial(C, 1) == pytest.approx(0.0, abs=1e-6)
    assert characteristic_polynomial(C, 2) == pytest.approx(0.0, abs=1e-6)
    assert characteristic_polynomial(C, 3) == pytest.approx(0.0, abs=1e-6)
    assert characteristic_polynomial(C, 0) == pytest.approx(6.0, abs=1e-6)  # det(C - 0I) = 1*2*3 = 6

def test_characteristic_polynomial_2x2():
    # Test with a 2x2 matrix
    C = Matrix([
        [4, 1],
        [2, 3]
    ])
    # Eigenvalues are 5 and 2
    assert characteristic_polynomial(C, 5) == pytest.approx(0.0, abs=1e-6)
    assert characteristic_polynomial(C, 2) == pytest.approx(0.0, abs=1e-6)
    assert characteristic_polynomial(C, 0) == pytest.approx(10.0, abs=1e-6)  # det(C) = 4*3 - 1*2 = 10

def test_characteristic_polynomial_3x3():
    # Test with a 3x3 matrix
    C = Matrix([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2]
    ])
    # One of the eigenvalues is 2
    assert characteristic_polynomial(C, 2) == pytest.approx(0.0, abs=1e-6)
    # det(C - 0I) = 4 (calculated separately)
    assert characteristic_polynomial(C, 0) == pytest.approx(4.0, abs=1e-6)

def test_characteristic_polynomial_edge_case():
    # Test with a matrix that has a zero determinant
    C = Matrix([
        [1, 2],
        [2, 4]
    ])
    # det(C) = 1*4 - 2*2 = 0, so λ=0 is an eigenvalue
    assert characteristic_polynomial(C, 0) == pytest.approx(0.0, abs=1e-6)
    # det(C - 5I) = (1-5)(4-5) - 2*2 = (-4)(-1) - 4 = 4 - 4 = 0
    # So λ=5 is also an eigenvalue (this matrix has eigenvalues 0 and 5)
    assert characteristic_polynomial(C, 5) == pytest.approx(0.0, abs=1e-6)

def test_find_eigenvalues_Jordan_2x2():
    C = Matrix([
        [2, 1],
        [0, 2]
    ])

    assert find_eigenvalues(C) == [2, 2]

def test_find_eigenvalues_Jordan_4x4():
    C = Matrix([
        [1,1,0,0],
        [0,2,0,0],
        [0,0,2,1],
        [0,0,0,3]
    ])
    assert find_eigenvalues(C) == [1, 2, 2, 3]

def test_find_eigenvalues_empty():
    C = Matrix([
        [],
        [],
        [],
        []
    ])
    assert find_eigenvalues(C) == []

def test_find_eigenvalues_NonDiag():

    C_1 = Matrix([
    [1, 2, 3],
    [0, 4, 5],
    [0, 0, 6]
])

    C_2 = Matrix([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

    assert find_eigenvalues(C_1) == [1, 4, 6]
    assert find_eigenvalues(C_2) == [-1, -1, 2] 
