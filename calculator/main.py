import numpy as np

if __name__ == "__main__":
    mat = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    mat1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    combine = mat[:, 1]
    combine = np.vstack((combine, mat1[:, 1]))
    print(combine)
