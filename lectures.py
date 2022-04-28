import math
import numpy as np

from matrixMath import MatrixMath

matrix = MatrixMath()


class Lectures():
    def __init__(self):
        x = 1

    def lecture60New(self):
        """Matrix asymmetry indes
        1. Return a measure of how symmetric a matrix is.
        2. Create a skew-symmetric matrix"""
        sM = MatrixMath.skewSymmetricMatrix(self, 3, -5, 5, "int")
        MatrixMath.printMatrix(self, sM)

    def lecture59New(self):
        """Conditions for a self-adjoint operator.
        The operator is a square symmetric matrix.
        Av dot w = v dot Aw.
        (Av)^T w = v^T A^T w = v^T Aw
        Because A^T = A: (Av)^T = v^TA
        v^T A w
        """
        v = matrix.randomMatrix(3, 1, -5, 5, int)
        w = matrix.randomMatrix(3, 1, -5, 5, int)
        #A = matrix.randomMatrix(3, 3, -5, 5, int)
        A = matrix.symetricMatrix(3, -5, 5, int)
        print(f"A = ")
        matrix.printMatrix(A)
        print("v = ")
        matrix.printMatrix(v)
        print("w =")
        matrix.printMatrix(w)

        print()
        # A dot v dot w
        B = matrix.matMult(A, v)
        #matrix.printMatrix(B)
        B = matrix.transpose(B)
        print(matrix.matMult(B, w))


        # w^T dot A dot v
        x = matrix.transpose(v)
        C = matrix.matMult(x, A)
        #C = matrix.transpose(C)
        #matrix.printMatrix(C)
        print(matrix.matMult(C, w))





    def lecture62(self):
        """Conditions for self-adjoint matrices
        <Av, w> = <v, Aw>
        v != w"""

    def lecture61(self):
        # A = matrix.enterMatrix()
        M = [[1, 2, 3], [4, 5, 6], [7, 7, 9]]
        A = np.array(M)

        # optional orthotona matrix to show that 2-norm is 1
        Q, r = np.linalg.qr(np.random.randn(5, 5))
        # A = Q

        # Frobenius norm
        normFrob = matrix.normFrobenius(A)

        normInd_2 = matrix.normInd2(A)
        # induced 2-norm
        # note computed as below
        lambd = np.sqrt(np.max(np.linalg.eig(A.T @ A)[0]))

        # schatten p-norm
        normSchatten = matrix.schatten_p_Norm(A, p=2)

        # % show all norms for coparison
        print(normFrob, normSchatten, '\n', normInd_2)

    def lecture60(self):
        A = matrix.randomMatrix(3, 3, -5, 5, int)
        B = matrix.randomMatrix(3, 3, -5, 5, int)
        size = matrix.frobeniusDotProductHadamard(A, B)
        print(size)

        size = matrix.frobeniusDotProductVectorize(A, B)
        print(size)

        size = matrix.frobeniusDotProductTrace(A, B)
        print(size)

        size = matrix.norm(A)
        print(size)

    def lecture59(self):
        import cmath
        import matplotlib.pyplot as plt

        n = 52
        # w is omega
        w = cmath.exp(-2 * 1j * math.pi / n)
        # print(f"w = {w}")

        # complex matrix
        F = matrix.complexMatrix(52, 52, 0, 0, 0, 0, float)
        # print(F)
        for j in range(n):
            for k in range(n):
                m = (j * k)
                F[j][k] = w ** m

        plt.imshow(F.real, cmap='jet')
        plt.show()
        plt.imshow(abs(F))
        plt.colorbar()
        # plt.show()

        x = np.random.randn(n)

        X1 = F @ x
        X2 = np.fft.fft(x)
        print(f'X1 == X2: {np.allclose(X1, X2)}')

        # plot them
        plt.figure()
        plt.plot(abs(X1))
        plt.plot(abs(X2), marker='o', color='r', linestyle='None')
        plt.show()
        plt.plot(F.real[:, 25])
        plt.show()

    def lecture58(self):
        A = matrix.randomMatrix(4, 4, -10, 10, int)
        SA = matrix.matMult(A, A)
        HA = matrix.hadamardMultiplication(A, A)
        print(f"SA = \n{SA}")
        print(f"HA = \n{HA}")
        D = matrix.diagMatrix(4, -10, 10, int)
        SD = matrix.matMult(D, D)
        HD = matrix.hadamardMultiplication(D, D)
        print(f"SD = \n{SD}")
        print(f"HD = \n{HD}")

    def lecture57(self):
        A = matrix.symetricToeplitzMatrix(3, -5, 5, int)
        B = matrix.symetricToeplitzMatrix(3, -5, 5, int)
        print(A)
        # B = A
        print(B)
        P = A * A
        print(f"P* = \n{P}")
        P = MatrixMath.hadamardMultiplication(self, A, B)
        print(f"PH = \n{P}")
        P = A @ B
        print(f"P@ = \n{P}")
        P = MatrixMath.matMult(self, A, B)
        print(f"PM = \n{P}")
        P = np.dot(A, B)
        print(f"Pdot = \n{P}")

    def lecture56(self):
        A = matrix.symetricMatrix(2, -4, 4, int)
        B = matrix.symetricMatrix(2, -4, 4, int)
        C = A + B
        D = matrix.matMult(A, B)
        E = matrix.hadamardMultiplication(A, B)
        print(A)
        print(B)
        print(f"sum = \n{C}")
        print(f"prod = \n{D}")
        print(f"hadamard = \n{E}")

    def lecture55(self):
        A = matrix.randomMatrix(3, 3, -2, 2, int)
        B = matrix.randomMatrix(3, 3, -2, 2, int)
        print(A)
        print(B)
        C = A * B
        print(C)
        C = matrix.hadamardMultiplication(A, B)
        print(C)

    def lecture54(self):
        A = matrix.randomMatrix(2, 2, -10, 10, int)
        S = matrix.symetrizeMatrix(A)
        # print(S)
        B = matrix.randomMatrix(3, 2, -3, 3, int)
        C = matrix.covarianceMatrix(B)
        print(f"C = {C}")

    def lecture52(self):
        # generate XY coordinates for a circle
        import matplotlib.pyplot as plt
        angles = []
        XY = []
        for angle in range(0, 360, 3):
            angle = angle * math.pi / 180
            XY.append([math.cos(angle), math.sin(angle)])
        print()
        R = matrix.randomMatrix(2, 2, -5, 5, int)
        print(R)
        R = matrix.symetrizeMatrix(R)
        R = matrix.singularizeAMatrix(R)
        print(f"R = {R}")
        for pt in XY:
            plt.plot(pt[0], pt[1], 'o')
            V = np.matrix(pt)

            T = matrix.transpose(V)

            z = matrix.matMult(R, T)
            plt.plot(z[0], z[1], 'o')

        plt.axis('square')
        plt.show()

    def liveEvil(self, A, B):
        res1 = np.transpose(A @ B)
        res2 = MatrixMath.transpose(self, B) @ MatrixMath.transpose(self, A)
        print(res1)
        print(res2)
        print(res1 - res2)