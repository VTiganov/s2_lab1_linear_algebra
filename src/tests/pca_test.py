import pytest
from linear_algebra import Matrix
from pca import center_data, covariance_matrix

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

