import math
import random

import numpy as np

from matrixMath import MatrixMath

matrix = MatrixMath()


class Lectures():
    def __init__(self):
        x = 1

    def lecture66(self):
        """Scalar multiplication and rank
        1. test whether matrix rank is invariant to scalar multiplication
        2. create 2 matrices: full rank R and a ruduced rank F, (both random)
        3. create scalar l
        4. print ranks of F, R l*f, l(R
        5. check wheter rank(l*F) == l*rank(F)
        """

        # 2. create 2 matrices: full rank R and a ruduced rank F, (both random)
        F = MatrixMath.randomMatrix(self, 3, 3, -5, 5, int)
        R = MatrixMath.randMatrixRowColRank(self, 3, 2, -5, 5, int)
        print("R =")
        MatrixMath.printMatrix(self, R)

        # 3. create scalar l
        l = random.randint(1, 10)
        print(f"l = {l}")
        lR = MatrixMath.scalarMultiplication(self, l, R)

        # 4 print ranks of F, R, l*F, l*rand(F)
        print(f"F rank = {MatrixMath.matrixRank(self, F)}")
        print(f"R rank = {MatrixMath.matrixRank(self, R)}")
        print(f"rank(lR) = {MatrixMath.matrixRank(self, lR)}")
        print(f"l*R rank(R) = {l * MatrixMath.matrixRank(self, R)}")



    def lecture65(self):
        """Create reduced-rank matrices using matrix multiplication
        Create a 10x10 matrix with rnak = 4 (using matrix multiplication)
        Generalize the procedure to create any MxN matrix with rank r"""
        A = MatrixMath.randomMatrix(self, 10, 10, -10, 10, int)
        B = MatrixMath.rankReducingMatrix(self, 10, 4)
        MatrixMath.printMatrix(self, B)
        print()
        C = MatrixMath.matMult(self, B, A)
        MatrixMath.printMatrix(self, A)
        print()
        MatrixMath.printMatrix(self, C)

        print(f"rankA = {MatrixMath.matrixRank(self, A)}")
        print(f"rankC = {MatrixMath.matrixRank(self, B)}")

        A = MatrixMath.randomMatrix(self, 10, 4, -5, 5, int)
        B = MatrixMath.randomMatrix(self, 4, 10, -5, 5, int)
        C = MatrixMath.matMult(self, A, B)
        print(f"rankC = {MatrixMath.matrixRank(self, B)}")

        MatrixMath.randMatrixRowColRank(self, 10, 4, -5, 5, int)

    def lecture64(self):
        """Rank of added and matrices"""
        print("rank of added matrices")
        A = [[1, 2, 3],[3, 4, 1],[5, 9, 1]]
        B = [[0, 3, 5],[1, 0, 4],[3, 3, 0]]
        C = A + B
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[1, 1, 1],[2, 2, 2],[3, 3, 0]]
        B = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        C = A + B
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[1, 2, 0],[3, 4, 0],[5, 9, 0]]
        B = [[0, 0, 5],[0, 0, 4],[0, 0, 1]]
        C = A + B
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[-1, -4, 2],[-4, 2, -1],[9, 4, -3]]
        B = [[1, 4, 0],[4, -2, 0],[-9, -4, 0]]
        C = A + B
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        print("rank of multiplied matrices")
        A = [[1, 2, 3],[3, 4, 1],[5, 9, 1]]
        B = [[0, 3, 5],[1, 0, 4],[3, 3, 0]]
        C = MatrixMath.matMult(self, A, B)
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[1, 1, 1],[2, 2, 2],[3, 3, 0]]
        B = [[0, 0, 0],[0, 0, 0],[0, 0, 0]]
        C = MatrixMath.matMult(self, A, B)
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[1, 2, 0],[3, 4, 0],[5, 9, 0]]
        B = [[0, 0, 5],[0, 0, 4],[0, 0, 1]]
        C = MatrixMath.matMult(self, A, B)
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()

        A = [[-1, -4, 2],[-4, 2, -1],[9, 4, -3]]
        B = [[1, 4, 0],[4, -2, 0],[-9, -4, 0]]
        C = MatrixMath.matMult(self, A, B)
        print(f"rank A: {MatrixMath.matrixRank(self, A, )}")
        print(f"rank B: {MatrixMath.matrixRank(self, B, )}")
        print(f"rank C: {MatrixMath.matrixRank(self, C, )}")
        print()


    def lecture63(self):
        """Methods to compute renk
        1. Count the number of columns in a linearly independent set.
        2. Apply row reduction to reduce matrix to echelon form and count number of pivots.
        3. Compute the singular value decomposition and count the number of non-zero singular values.
        4. Compute the eigendecomposition and count the number of non-zero eigenvalues.
        """
        m,n = 4, 6
        # Create a random matrix
        A = np.random.randn(m, n)
        MatrixMath.printMatrix(self, A)

        # What is the largest possible rank?
        rank = np.linalg.matrix_rank(A)
        print(f"rank = {rank}")

        #Make last column same as second to last column.
        #No change in rank
        B = A
        B[:, -1] = B[:, -2]
        MatrixMath.printMatrix(self, B)
        rank = np.linalg.matrix_rank(B)
        print(f"rank = {rank}")

        #Make last row same as next to last row.
        #Rank ruduced by 1.
        B = A
        B[-1,:] = B[-2,:]
        MatrixMath.printMatrix(self, B)
        rank = np.linalg.matrix_rank(B)
        print(f"rank = {rank} \n")

        A = np.random.randn(m, m)
        B = A
        print("A")
        MatrixMath.printMatrix(self, A)
        print(f"rank = {np.linalg.matrix_rank(A)} \n")

        #Column 3 = column 4 in column B
        B[:, -1] = B[:, -2]
        print("B")
        MatrixMath.printMatrix(self, B)
        print(f"rank = {np.linalg.matrix_rank(B)} \n")

        print(f"B =")
        #noise level: add small amount of noise to reduced rank matrix
        noiseLevel = 0.000000000000001

        #add noise to matrix B
        B = B + noiseLevel*np.random.randn(m,m)
        MatrixMath.printMatrix(self, B)
        print(f"rank = {np.linalg.matrix_rank(B)} \n")


    def lecture62(self):
        """Rank give information about the amount of information contained in a matrix.
        6 Things to know about rank
        1. r or rank(A). A single number for a matrix.
            rank is a single non-negative integer. Related to the dimension of a matrix.
        2. max(r) = min(m,n). Max rank is the smaller of number of rows or columns.
            r is an element of N, s.t. (such that) 0<= r <= min{m,n}
        3. rank is a property of the entire matrix.
            No such thing as the rank of the column space or the row space.
        4. Max rank(A) mxm = m  "Full rank matrix" if less Degenerate, Singular, Non invertible matrix
           max rank(A) m>n = n  "Full column rank"
           max rank(A) m<n = m  "full row rank matrix
           If not full then it is "Degenerate"  "Rank Deficient"  "Reduced Rank" Low rank"
        5. Rank = dimensionality of information in the matrix.
        6. Rank One def: is the largest number of columns or rows that can form a linearly independent set.
        """

    def lecture61New(self):
        """Matrix Division
        1. Dadamard element by element division"""
        A = MatrixMath.randomMatrix(self, 3, 3, -10, 10, int)
        B = MatrixMath.randomMatrix(self, 3, 3, 1, 10 , int)
        C = MatrixMath.hadamarDivision(self, A, B)
        MatrixMath.printMatrix(self, A)
        print()
        MatrixMath.printMatrix(self, B)
        print()
        MatrixMath.printMatrix(self, C)


    def lecture60New(self):
        """Matrix asymmetry index
        1. Return a measure of how symmetric a matrix is.
        2. Create a skew-symmetric matrix"""
        A = MatrixMath.randomMatrix(self, 3, 3, -5, 5, int)
        A = MatrixMath.symetricMatrix(self, 3, -5, 5, int)
        A = MatrixMath.skewSymmetricMatrix(self, 3, -5, 5, int)
        #MatrixMath.printMatrix(self, A)
        #print(MatrixMath.norm(self, A))
        #print(MatrixMath.matrixTilda(self,  A))
        asymmetry = MatrixMath.asymmetryIndex(self, A)
        #print(f"Asymmetry = \n {asymmetry} \n \n")

        """Create matrix with particular matrix index
        1. Make symmetric matrix; S
        2. Make asymmetrix matrix; K
        3. Generate a ranging from 0 to 1.
        4. Multiply S by a and K by (a-1)
        5. 
        """
        asymmetry_index = []
        for i in range (11):
            a = i/10
            S = (1-a)*MatrixMath.symetricMatrix(self, 3, 0.0, 10.0, float)
            K = (a)*MatrixMath.skewSymmetricMatrix(self, 3, 0.0, 10.0, float)
            A = S + K
            asymmetry = MatrixMath.asymmetryIndex(self, A)
            asymmetry_index.append([a, asymmetry])
        print(f"assymetry = {asymmetry_index}")

        A = MatrixMath.randomMatrix(self, 3, 3, -10, 10, int)
        MatrixMath.printMatrix(self, A)
        print()
        S = MatrixMath.symmetricMatricFromA(self, A)
        MatrixMath.printMatrix(self, S)
        print()

        K = MatrixMath.skewSymmetricMatrixFromA(self, A)
        MatrixMath.printMatrix(self, K)

        Ai = []
        for i in range (11):
            p = i/10
            C = p*K + (1-p)*S
            asymmetry = MatrixMath.asymmetryIndex(self, C)
            Ai.append([p, asymmetry])
        print(Ai)


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