from linear_algebra import Matrix
import pytest

def test_addition():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    result = A.add(B)
    expected = [[6, 8], [10, 12]]
    assert result.data == expected

def test_scalar_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    result = A.multiply(2)
    expected = [[2, 4], [6, 8]]
    assert result.data == expected

def test_matrix_multiplication():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[2, 0], [1, 2]])
    result = A.multiply(B)
    expected = [[4, 4], [10, 8]]
    assert result.data == expected

def test_transpose():
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    result = A.transpose()
    expected = [[1, 4], [2, 5], [3, 6]]
    assert result.data == expected

def test_gauss_solve_simple():
    A = Matrix([[2, 1], [5, 7]])
    b = Matrix([[11], [14]])
    result = Matrix.gauss_solve(A, b)
    # Ожидаем решение: x = 7, y = -3
    expected = [[7.0], [-3.0]]
    for r, e in zip(result.data, expected):
        assert abs(r[0] - e[0]) < 1e-6

def test_gauss_solve_inconsistent():
    A = Matrix([[1, 1], [2, 2]])
    b = Matrix([[3], [8]])  # нет решения
    with pytest.raises(ValueError, match="Система не имеет решения"):
        Matrix.gauss_solve(A, b)

def test_determinant_zero():
    A = Matrix([[1, 1], [1, 1]])
    assert Matrix.determinant(A) == 0

def test_determinant_nonzero():
    A = Matrix([[1, 2], [3, 4]])
    assert Matrix.determinant(A) == -2
