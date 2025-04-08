class Matrix:
    def __init__(self, data):
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        self.data = data

    def print_matrix(self):
        for row in self.data:
            print(' '.join(f'{x:.2f}' for x in row))

    def add(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Матрицы должны быть одного размера")
        result = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def multiply(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[x * other for x in row] for row in self.data])
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Количество столбцов первой матрицы должно совпадать с количеством строк второй")
            result = [
                [sum(a * b for a, b in zip(row, col)) 
                for col in zip(*other.data)]
                for row in self.data
            ]
            return Matrix(result)

    def transpose(self):
        return Matrix([list(col) for col in zip(*self.data)])
    
    def swap_rows(self, i, j):
        self.data[i], self.data[j] = self.data[j], self.data[i]
    
    def get_column(self, j):
        return [row[j] for row in self.data] 
       
    

    @staticmethod
    def gauss_solve(A, b):
    
        n = A.rows
        if n != A.cols or n != b.rows or b.cols != 1:
            raise ValueError("Неверные размеры матриц")
    
        augmented = [row.copy() for row in A.data]
        for i in range(n):
            augmented[i].append(b.data[i][0])
    
        for col in range(n):
            max_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
            augmented[col], augmented[max_row] = augmented[max_row], augmented[col]
        
            pivot = augmented[col][col]
            if abs(pivot) < 1e-10:
                raise ValueError("Система не имеет решения")
        
            augmented[col] = [x / pivot for x in augmented[col]]
        
            for row in range(col + 1, n):
                factor = augmented[row][col]
                augmented[row] = [
                    augmented[row][i] - factor * augmented[col][i]
                    for i in range(n + 1)
                ]
    
        solution = [0] * n
        for i in range(n-1, -1, -1):
            solution[i] = augmented[i][-1] - sum(
                augmented[i][j] * solution[j] for j in range(i + 1, n)
            )
        
        return Matrix([[x] for x in solution])


