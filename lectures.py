import math
import random

import numpy as np

from matrixMath import MatrixMath

matrix = MatrixMath()


class Lectures():
    def __init__(self):
        x = 1


class Section6():
    """Matrix rank"""

    def lecture70(self):
        """Is this vector in the span of this set
        Determine whether this vector"""
        V = [[1, 2, 3, 4]] #The inner bracket creates the list. The outer bracket tells python that this is a matrix.
        V = MatrixMath.transpose(self, V)
        """is in the span of these sets"""
        S =  [[4, 3, 6, 2], [0, 4, 0, 1]]
        S = MatrixMath.transpose(self, S)

        T = [ [1, 2, 2, 2], [0, 0, 1, 2] ]
        T = MatrixMath.transpose(self, T)

        C = MatrixMath.concatenateMatrices(self, S, V)
        print("The rank of V + S is greater than S, therefore V is not in the span of S")

        D = MatrixMath.concatenateMatrices(self, T, V)
        print("The rank of V + T is equal to the rank of T, therefore V is in the span of T")

        rankV = MatrixMath.matrixRank(self, V)
        rankS = MatrixMath.matrixRank(self, S)
        rankC = MatrixMath.matrixRank(self, C)
        rankT = MatrixMath.matrixRank(self, T)
        rankD = MatrixMath.matrixRank(self, D)

        print(f"rank(V) = {rankV}, rank(S) = {rankS}, rank(S + V) = {rankC}")
        print(f"rank(V) = {rankV}, rank(T) = {rankT}, rnak(T + V) = {rankD}")

    def lecture69(self):
        """Making a matrix full rank by "shifting"
        1. See the effects of "shifting" a matrix by lambdaI.
        2. Appreciate the difficulty of knowing the right amounbt of shifting
        A_Tilda = A + lambdaI       lambda is a scalar
        A_Tilda, A and lambdaI must be square matrices
        """

        #Shifting a matrix: extreme example
        """Zero matrix plus identity matrix"""
        A = MatrixMath.zeroMatrix(self, 3)
        I = MatrixMath.identityMatrix(self, 3)
        C = MatrixMath.addMatrices(self, A, I)
        rankA = MatrixMath.matrixRank(self, A)
        rankI = MatrixMath.matrixRank(self, I)
        rankC = MatrixMath.matrixRank(self, C)
        print(f"rank(A) = {rankA}, rank(I) = {rankI}, rank(C) = {rankC}")

        A = [[1, 3, -19], [5, -7, 59], [-5, 2, -24]]
        MatrixMath.printMatrix(self, A)
        print(f"rankA = {MatrixMath.matrixRank(self, A)}")
        l = 0.01
        lambdaI = l * MatrixMath.identityMatrix(self, 3)
        MatrixMath.printMatrix(self, lambdaI)
        print(f"rank(lambdaI) = {MatrixMath.matrixRank(self, lambdaI)}")
        C = MatrixMath.addMatrices(self, A, lambdaI)
        MatrixMath.printMatrix(self, C)
        print(f"rank(C) = {MatrixMath.matrixRank(self, C)}")
        print()

        m, n = 30, 4
        A = MatrixMath.randomMatrix(self, m, n, -5, 5, int)
        B = MatrixMath.randomMatrix(self, n, m, -5, 5, int)
        C = A@B
        rankC = MatrixMath.matrixRank(self, C)
        #MatrixMath.printMatrix(self, C)
        print(f"Rank(C) = {rankC}")

        # shift amount (l = lambda)
        l = 0.01

        #new matrix
        B = C + l*MatrixMath.identityMatrix(self, m)
        rankB = MatrixMath.matrixRank(self, B)
        print(f"rank(B) = {rankB}")

    def lecture68(self):
        """Rank of multiplied and summed matrices.
        Rules: rank of AB<=min( rank(A), rank(B) )
               rank of A+B<=rank(A) + rank(B)

        Generate 2 matrices (A and B), 2x5
        compute ATxA and BTxB
        Find the ranks of ATxA and BTxB
        Find the rank of ATxAxBTxB
        Find the rank of ATxA + BTxB
        At start of each section predict the outcome of each part"""

        #Generate 2 matrices (A and B), 2x5
        A = MatrixMath.randomMatrix(self, 2, 5, -3, 3, int)
        B = MatrixMath.randomMatrix(self, 2, 5, -3, 3, int)

        #Compute ATxA and BTxB
        AT = MatrixMath.transpose(self, A)
        BT = MatrixMath.transpose(self, B)
        ATxA = MatrixMath.matMult(self, AT, A)
        BTxB = MatrixMath.matMult(self, BT, B)

        #Find the ranks of ATxA and BTxB
        """ATxA and BTxB will have dimensions of 5x5 and will be of rank 2"""
        sizeATxA = MatrixMath.size(self, ATxA)
        sizeBTxB = MatrixMath.size(self, BTxB)
        rankATxA = MatrixMath.matrixRank(self, ATxA)
        rankBTxB = MatrixMath.matrixRank(self, BTxB)
        print(f"sizeATxA = {sizeATxA[0]}x{sizeATxA[1]}, rankATxA = {rankATxA}")
        print(f"sizeBTxB = {sizeBTxB[0]}x{sizeBTxB[1]}, rankBTxB = {rankBTxB}")

        #Find the ranks of ATxAXBTxB
        """The rank of ATxAXBTxB will be 2"""
        ATxAXBTxT = MatrixMath.matMult(self, ATxA, BTxB)
        sizeATxAXBTxT = MatrixMath.size(self, ATxAXBTxT)
        rankATxAXBTxT = MatrixMath.matrixRank(self, ATxAXBTxT)
        print(f"sizeATxAXBTxT = {sizeATxAXBTxT[0]}x{sizeATxAXBTxT[1]}, rankATxAXBTxT = {rankATxAXBTxT}")

        #Find the rank of ATxA + BTxB
        """The rank of ATxA + BTxB will be up to 4. I could be 2, 3, or 4 depending on which rows or columns are
        independent"""
        ATxAPlusBTxB = MatrixMath.addMatrices(self, ATxA, BTxB)
        sizeATxAPlusBTxB = MatrixMath.size(self, ATxAPlusBTxB)
        rankATxAPlusBTxB = MatrixMath.matrixRank(self, ATxAPlusBTxB)
        print(f"sizeATxAPlusBTxB = {sizeATxAPlusBTxB[0]}x{sizeATxAPlusBTxB[1]}, rankATxAPlusBTxB = {rankATxAPlusBTxB}")



    def lecture67(self):
        """Rank of Atransposa * A and A * A transpose
        rank(A) = rank(A^T * A) = rand(A^T) = rank(A * A^T)"""

        #A = MatrixMath.randomMatrix(self, 3, 3, 0, 5, int)
        A = MatrixMath.randMatrixRowColRank(self, 3, 2, 0, 2, int)
        T = MatrixMath.transpose(self, A)
        rankA = MatrixMath.matrixRank(self, A)
        rankT = MatrixMath.matrixRank(self, T)
        MatrixMath.printMatrix(self, A)
        print(f"rankA = {rankA}")
        MatrixMath.printMatrix(self, T)
        print(f"rankT = {rankT}")

        ATxA = MatrixMath.matMult(self, T, A)
        AxAT = MatrixMath.matMult(self, A, T)
        rankATxA = MatrixMath.matrixRank(self, ATxA)
        rankAxAT = MatrixMath.matrixRank(self, AxAT)
        MatrixMath.printMatrix(self, ATxA)
        print(f"rankATxA = {rankATxA}")
        MatrixMath.printMatrix(self, AxAT)
        print(f"rankAxAT = {rankAxAT}")

        m, n = 14, 3
        A = MatrixMath.randomMatrix(self, m, n, 0, 5, int)
        T = MatrixMath.transpose(self, A)
        ATxA = MatrixMath.matMult(self, T, A)
        AxAT = MatrixMath.matMult(self, A, T)
        size_ATxA = MatrixMath.size(self, ATxA)
        size_AxAT = MatrixMath.size(self, AxAT)
        rankATxA = MatrixMath.matrixRank(self, ATxA)
        rankAxAT = MatrixMath.matrixRank(self, AxAT)

        print(f"ATxA: {size_ATxA[0]}x{size_ATxA[1]}, rank = {rankATxA}")
        print(f"AxAT: {size_AxAT[0]}x{size_AxAT[1]}, rank = {rankAxAT}")

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

class Section5():
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