from linear_algebra import Matrix

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

