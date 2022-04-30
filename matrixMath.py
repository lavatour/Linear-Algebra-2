import random

import numpy as np
import math

"""TO DO
LinearDependence
LinearIndependence
Span
SubSpace
SubSet
IsSubspaceOrSubset
IsBasis There can be only one combination of the vectors for it to be a basis.
IndependentComponentAnalysis
PrincipalComponentAnalysis
ColumnSpaceR(M)
RowSpaceC(N)
Tensors
MatrixTyps: Square, Rectangular, Symetric(mirrored across diagonal) SkewSymetric 
ScalarIdentity
"""


class MatrixMath():
    """
    def unitVector(self, V):

    MAKE MATRICES
    unitVector(2dVector):
    randomMatrix(rows, cols, max min type)
    complexMatrix(rows, cols, rMin, rMax, iMin, Imax)
    identityMatrix(size)
    zeroMatrix(rows, cols)
    rotationMatrix2x2(angle)
    diagMatrix(self, rows, min, max, intOrFloat)
    symetricMatrix(size, min, max, type, type)
    upperTriangularMatrix(size, min, max, type)
    lowerTriangularMatrix(size, min, max, type)
    diagonal(A) Returns a diagonal matrix

    MATRIX OPERATIONS
    addMatrices(A, B)
    subtractMatrices(A, B)
    scalarMultiplication(s, A)
    transpose(A)
    hermitianTranspose(A) for complex matrices
    dotProduct(A, B)
    matMult(A,B)
    columnDotProduct(A, B)
    outerProduct(A, B)
    crossProduct(A, B) 3D vectors only
    hadamardMultiplication(A, B) element by element
    trace(A) returns the trace of matrix A
    rowAddBroadcasting(A, r) adds row r to each row
    rowSubtractBroadcasting(A, r)
    rowMultiplyBroadcasting(A, r)
    rowDivisionBroadcasting(A, r)
    colAddBroadcasting(A, r)
    colSubtractBroadcasting(A, r)
    colMultBroadcasting(A, r)
    colDivideBroadcasting(A, r)
    diaganolOfMatrix(A)

    MATRIX INFORMATION DATA
    size(A) rows, cols
    vectorLength(V)
    angleBetweenVectors(V1, V2)
    isComplex(A)
    complexVectorSize(A)
    concatenateMatrices(A, B)
    matrixDataType(A)
    isSymetric

    DISPLAY
    printMatrix
    displayVector2D
    displayVector3D
    displayVector3D-1
    """

    def __init__(self):
        self.a = 1

    def enterMatrix(self):
      print("enter the size rows & columns or mxn")
      rows = int(input("enter rows."))
      cols = int(input("enter columns"))
      A = MatrixMath.zeroMatrix(self, rows)
      print("Creating zero matrix")
      print(A)
      print("replace each element, now a zero.")
      for row in range(rows):
        for col in range(cols):
          A[row][col] = int(input(f"Replace element {row+1},{col+1}  "))
          print(A)
      return A

    def unitVector(self, V):
        """Converts 2d vector to unit vector of length 1"""
        v1 = np.array(V)
        v1 = np.ndarray.tolist(v1)
        #print(v1)
        mV1 = MatrixMath.vectorLength(self, v1)
        u1 = MatrixMath.scalarMultiplication(self, 1 / mV1, v1)
        MatrixMath.displayVector2D(self, v1, u1)
        return u1

    def printMatrix(self, A):
        [rows, cols] = MatrixMath.size(self, A)
        if rows > 1:
            for row in range(rows):
                print(A[row])
        else:
            print(A)

    def randomMatrix(self, rows, cols, minVal, maxVal, numTyp):
        """Create matrix elements of type float between minVal and maxVal."""
        A = np.zeros((rows, cols))
        # print(type(A[0][0]))
        for row in range(rows):
            for col in range(cols):
                if numTyp == int:
                    A[row][col] = random.randint(minVal, maxVal)
                if numTyp == float:
                    A[row][col] = random.randint(minVal, maxVal) * random.random()
        return A

    def complexMatrix(self, rows, cols, rMin, rMax, iMin, iMax, numTyp):
        """Makes a complex matrix rows X cols. Real from rMin to rMax, imaginary from iMin to iMax."""
        A = np.zeros((rows, cols), dtype=complex)
        C = A.astype(complex)
        for row in range(rows):
            for col in range(cols):
                if numTyp == int:
                    real = random.randint(rMin, rMax)
                    imag = random.randint(iMin, iMax)
                if numTyp == float:
                    real = random.randint(rMin, rMax) * random.random()
                    imag = random.randint(iMin, iMax) * random.random()
                comp = np.complex(real, imag)
                C[row][col] = comp
        return C

    def identityMatrix(self, m):
        I = np.eye(m)
        return I

    def zeroMatrix(self, m):
        """dataType can be integer, float, or complex.
        By default the datatype is float."""
        Z = np.zeros((m, m), float)
        return Z

    def rotationMatrix2x2(self, theta, unit="rad"):
        print(theta)
        if unit == "deg" or unit == "degree":
            theta = theta * math.pi / 180
            unit = "rad"
        if unit == "radian":
            unit = "rad"
        if unit != "rad":
            "Rotation Matrix method: unit not correctly defined."
            return
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        return R

    def diagMatrix(self, rows, minVal, maxVal, intOrFloat):
        A = MatrixMath.zeroMatrix(self, rows)
        for row in range(rows):
            for col in range(rows):
                if row == col:
                    if intOrFloat == float:
                        A[row][col] = random.randint(minVal, maxVal) * np.random.random()
                    elif intOrFloat == int:
                        A[row][col] = random.randint(minVal, maxVal)
        return A

    def symmetricMatricFromA(self, A):
        """Returns the symmetric part of matrix A as defined by:
        S = (A + A_transpose) / 2"""
        S = (A + MatrixMath.transpose(self, A))/2
        return S

    def symetricMatrix(self, size, minVal, maxVal, intOrFloat):
        A = np.zeros((size, size), intOrFloat)
        for row in range(size):
            for col in range(size):
                if col <= row:
                    A[row][col] = random.randint(minVal, maxVal)
                    A[col][row] = A[row][col]
        return A

    def skewSymmetricMatrixFromA(self, A):
        """Returns the asymmetric part of matrix A as defined by:
        K = A - A_transpose"""
        K = (A - MatrixMath.transpose(self, A)) / 2
        return K

    def skewSymmetricMatrix(self, size, minVal, maxVal, intOrFloat):
        A = np.zeros((size, size), intOrFloat)
        for row in range(size):
            for col in range(size):
                if col < row:
                    A[row][col] = random.randint(minVal, maxVal)
                    A[col][row] = -1 * A[row][col]
        return A

    def upperTriangularMatrix(self, size, minVal, maxVal, dataTyp=float):
        M = MatrixMath.randomMatrix(self, size, size, minVal, maxVal, dataTyp)
        U = np.triu(M)
        return U

    def lowerTriangularMatrix(self, size, minVal, maxVal, dataTyp=float):
        M = MatrixMath.randomMatrix(self, size, size, minVal, maxVal, dataTyp)
        L = np.tril(M)
        return L

    def singularMatrix(self, size, min, max):
        A = MatrixMath.randomMatrix(self, size, size, min, max, numTyp=int)
        numbers = []
        for num in range(size):
            numbers.append(num)
        s = numbers.pop(random.randint(0, size - 1))
        r = numbers.pop(random.randint(0, size - 2))
        const = random.randint(2, 10)
        for row in range(size):
            A[row][r] = const * A[row][s]
        # print(A)
        return A

    def singularizeAMatrix(self, A):
        [rows, cols] = MatrixMath.size(self, A)
        if rows != cols:
            print("Cannot be a singular matrix. not square.")
            return
        numbers = []
        for num in range(rows):
            numbers.append(num)
        s = numbers.pop(random.randint(0, rows - 1))
        r = numbers.pop(random.randint(0, rows - 2))
        const = random.randint(2, 10)
        for row in range(rows):
            A[row][r] = const * A[row][s]
        print(type(A))
        if type(A) != list:
            A = A.tolist()
        return A

    def symetrizeMatrix(self, A):
        S = (A + MatrixMath.transpose(self, A)) / 2
        # print(A)
        # print(S)
        return S

    def symetricToeplitzMatrix(self, size, min, max, intOrFloat = int):
        """Symmetric matrix with constant diagonal"""
        A = MatrixMath.symetricMatrix(self, size, min, max, intOrFloat)
        #print(f"A = \n{A}")
        if intOrFloat == int:
            diag = random.randint(min, max)
        else:
            diag = random.random(min, max)
        for row in range(size):
            A[row][row] = diag
        return A

    def rankReducingMatrix(self, size, rank):
        """Reduce a matrix from maximum rank size to maximum rank "rank"""
        A = np.zeros((size, size))
        for row in range(size):
            for col in range(size):
                if col < rank:
                    A[row][col] = random.randint(-10, 10)
        return A

    def randMatrixRowColRank(self, size, rank, min, max, dataType = int):
        A = MatrixMath.randomMatrix(self, size, rank, -5, 5, int)
        B = MatrixMath.randomMatrix(self, rank, size, -5, 5, int)
        C = MatrixMath.matMult(self, A, B)
        return C


    """MATRIX OPERATIONS"""

    def addMatrices(self, A, B):
        """A + B"""
        if type(A) == list:
            np.asarray(A)
        if type(B) == list:
            np.asarray(B)
        [rows, cols] = MatrixMath.size(self, A)
        [rowB, colB] = MatrixMath.size(self, B)
        C = A + B
        return C

    def subtractMatrices(self, A, B):
        """A - B"""
        if type(A) == list:
            np.asarray(A)
        if type(B) == list:
            np.asarray(B)
        C = A - B
        return C

    def scalarMultiplication(self, scalar, A):
        """Multiply a matrix by a scalar"""
        if type(A) == list:
            np.asarray(A)
        s = scalar
        B = s * A
        return B

    def size(self, A):
        """Return matrix size"""
        rows = len(A)
        cols = len(A[0])
        return [rows, cols]

    def displayVector2D(self, V1=[0, 0], V2=[0, 0], V3=[0, 0], V4=[0, 0]):
        import matplotlib.pyplot as plt
        v = np.array(V1)
        u = np.array(V2)
        plt.plot([0, v[0]], [0, v[1]], 'b', label='V')
        plt.plot([0, u[0]], [0, u[1]], 'r', label='V')
        plt.axis('square')
        axlim = int(max(abs(V1)) * 1.6)
        # print(max(abs(V1)))
        plt.axis((-axlim, axlim, -axlim, axlim))
        plt.grid
        plt.show()

    def displayVector3D(self, v1=[0, 0, 0], v2=[0, 0, 0], v3=[0, 0, 0], v4=[0, 0, 0]):
        import matplotlib.pyplot as plt
        lenV1 = MatrixMath.vectorLength(self, v1)
        lenV2 = MatrixMath.vectorLength(self, v2)
        lenV3 = MatrixMath.vectorLength(self, v3)
        lenV4 = MatrixMath.vectorLength(self, v4)
        axis = int(max(lenV1, lenV2, lenV3, lenV4) * 1.5)  # Sets the axis length based on vector size
        # v1X, v1Y, v1Z = v1[0], v1[1], v1[2]
        print(f"axis = {axis}")
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # draw vectors
        ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], 'r', linewidth=3)
        ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], 'r', linewidth=3)
        ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]], 'g', linewidth=3)
        ax.plot([0, v4[0]], [0, v4[1]], [0, v4[2]], 'r', linewidth=3)

        xx, yy, = np.meshgrid(range(-15, 16), range(-15, 16))
        cp = np.cross(v1, v2)
        z1 = (-cp[0] * xx - cp[1] * yy)
        ax.plot_surface(xx, yy, z1)
        # plt.axis((-axis, axis, -axis, axis))
        # plt.title('Angle between vectors: %s rad.')
        plt.show()

    def transpose(self, A):
        #print(A)
        """Return the transpose matrix"""

        if type(A) == list:
            A = np.matrix(A)
        T = np.transpose(A)
        #T = T.tolist()
        #MatrixMath.printMatrix(self, T)
        return T

    def hermitianTranspose(self, A):
        T = np.transpose(A)
        CT = np.conjugate(T)
        return CT

    def dotProd(self, A, B):
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        # print(f"rowsA, rowsB = {colsA, colsB}")
        if colsA != colsB:
            print("Error, Dot product impossible, matrices different sizes")
            return
        A = np.array(A)
        B = np.array(B)
        dotProd = np.dot(A, B)
        return dotProd

    def matMult(self, A, B):
        if type(A) == list:
            A = np.array(A)
        if type(B) == list:
            B = np.array(B)
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if colsA != rowsB:
            print("Error, Dot product impossible, matrices incorrect sizes")
            return
        dotProd = np.matmul(A, B)
        return dotProd

    def columnDotProduct(self, A, B):
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if colsA > 1:
            dP1 = []
        else:
            dP1 = 0.0
        if colsA != colsB or rowsA != rowsB:
            print("Columns not equal length cannot perform column dot product")
            return
        for col in range(colsA):
            sum = 0
            for row in range(rowsA):
                sum += A[row][col] * B[row][col]
            if type(dP1) == float:
                dP1 = sum
                return dP1
            else:
                dP1.append(sum)
        return dP1

    def hadamardMultiplication(self, A, B):
        """Element by element multiplication"""
        C = []
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if [rowsA, colsA] != [rowsB, colsB]:
            print("Hadamard multiplication impossible, different sizes")
            return
        for row in range(rowsA):
            C.append([])
            for col in range(colsA):
                C[row].append(A[row][col] * B[row][col])
        #C = np.matrix(C)
        return C

    def hadamarDivision(self, A, B):
        """Element by element division"""
        C = []
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if [rowsA, colsA] != [rowsB, colsB]:
            print("Hadamar division impossible, matrices A and B are of different sizes.")
            return
        for row in range(rowsA):
            C.append([])
            for col in range(colsA):
                C[row].append((A[row][col] / B[row][col]))
        return C

    def outerProduct(self, A, B):
        """Outer product A * B^T """
        T = MatrixMath.transpose(self, B)
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if rowsA > 1 or colsB < 1:
            print("These are not matrices. Outer Product impossible")
            return
        outerProd = np.outer(A, B)
        return outerProd

    def crossProduct(self, A, B):
        """Cross product of 2 3D vectors
        returns a normal vector"""
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if rowsA == rowsA == 1 and colsA == colsB == 3:
            crossProd = np.cross(A, B)
            return crossProd

    def concatenateMatrices(self, A, B):
        sizeA = MatrixMath.size(self, A)
        sizeB = MatrixMath.size(self, B)
        rowsA = rowsB = sizeA[0], sizeB[0]
        if rowsA != rowsB:
            print("Concatenation not possible. The number of rows in A must be the same as the number of rows in B")
            return
        C = np.concatenate((A, B), axis=1)
        return C

    def diagonalOfMatrix(self, A):
        D = np.diag(A)
        return D

    def rowAddBroadcasting(self, A, r):
        """Used in AI"""
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, r)
        if colsM == colsV and rowsV == 1:
            B = A + r
            return B
        else:
            print("Vector and matrix dimensions not suitable for row broadcasting")
            return

    def rowSubtractBroadcasting(self, A, r):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, r)
        if colsM == colsV and rowsV == 1:
            B = A - r
            return B
        else:
            print("Vector and matrix dimensions not suitable for row broadcasting")
            return
        return

    def rowMultiplyBroadcasting(self, A, r):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, r)
        if colsM == colsV and rowsV == 1:
            B = A * r
            return B
        else:
            print("Vector and matrix dimensions not suitable for row broadcasting")
            return

    def rowDivideBroadcasting(self, A, r):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, r)
        if colsM == colsV and rowsV == 1:
            B = A / r
            return B
        else:
            print("Vector and matrix dimensions not suitable for row broadcasting")
            return

    def colAddBroadcasting(self, A, c):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, c)
        if rowsM == rowsV and colsV == 1:
            B = A + np.reshape(c, (len(c), 1))
            return B
        else:
            print("Vector and matrix dimensions not suitable for column broadcasting")
            return

    def colSubtractBroadcasting(self, A, c):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, c)
        if rowsM == rowsV and colsV == 1:
            B = A - np.reshape(c, (len(c), 1))
            return B
        else:
            print("Vector and matrix dimensions not suitable for column broadcasting")
            return

    def colMultBroadcasting(self, A, c):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, c)
        if rowsM == rowsV and colsV == 1:
            B = A * np.reshape(c, (len(c), 1))
            return B
        else:
            print("Vector and matrix dimensions not suitable for column broadcasting")
            return

    def colDivideBroadcasting(self, A, c):
        [rowsM, colsM] = MatrixMath.size(self, A)
        [rowsV, colsV] = MatrixMath.size(self, c)
        if rowsM == rowsV and colsV == 1:
            B = A / np.reshape(c, (len(c), 1))
            return B
        else:
            print("Vector and matrix dimensions not suitable for column broadcasting")
            return

    def frobeniusDotProductHadamard(self, A, B):
        magnitude = 0
        [rowsA, colsA] = MatrixMath.size(self, A)
        [rowsB, colsB] = MatrixMath.size(self, B)
        if rowsA != rowsB or rowsB != colsB:
            print("forbenius dot product not possible. Matrices different sizes.")
            return
        H = MatrixMath.hadamardMultiplication(self, A, B)
        for row in range(rowsA):
            for col in range(colsA):
                #print(row, col)
                #print(H[row][col])
                magnitude += H[row][col]
        #print(magnitude)
        return magnitude

    def frobeniusDotProductVectorize(self, A, B):
        M1 = np.matrix(A)
        M2 = np.matrix(B)
        V1 = M1.flatten()
        V2 = M2.flatten()
        V2T = MatrixMath.transpose(self, V2)
        magnitude = V1 * V2T
        size = magnitude.tolist()
        return size[0][0]

    def frobeniusDotProductTrace(self, A, B):
        At = MatrixMath.transpose(self, A)
        M = At @ B
        trace = MatrixMath.trace(self, M)
        return trace


    """MATRIX INFORMATION"""
    def normFrobenius(self, A):
      normFrob = np.linalg.norm(A, 'fro')
      return normFrob

    def normInd2(self, A):
      Ind2Norm = np.linalg.norm(A, 2)
      return Ind2Norm

    def schatten_p_Norm(self, A, p = 2):
      # p = 2 but can be 1 etc
      s = np.linalg.svd(A)[1] # get singular values
      normSchatten_p = np.sum(s**p)**(1/p)
      return normSchatten_p

    def norm(self, A):
        A = np.array(A)
        At = A.transpose()
        AAt = A @ At
        normA =math.sqrt(MatrixMath.trace(self, AAt))
        return normA

    def vectorLength(self, A):
        """THIS IS INCOMPLETE & MAY HAVE TO BE CHANGED TO np.linalg.norm(v)"""
        dp = MatrixMath.dotProd(self, A, A)
        length = math.sqrt(dp)
        mag = np.linalg.norm(A)
        # print(mag, length)
        return length

    def angleBetweenVectors(self, A, B):
        dp = MatrixMath.dotProd(self, A, B)
        lenA = MatrixMath.vectorLength(self, A)
        lenB = MatrixMath.vectorLength(self, B)
        theta = math.acos(dp / (lenA * lenB))
        return theta

    def isComplex(self, A):
        #print(A)
        complx = np.iscomplex(A)
        return complx

    def covarianceMatrix(self, A):
        #print(A)
        At = np.transpose(A)
        #print(f"At = {At}")
        C = np.matmul(At, A)
        #print(C)
        return C

    def matrixTilda(self, A):
        """Returns the asymmetric part of matrix A
        ATilda = (A - ATranspose) / 2"""
        A_transpose = MatrixMath.transpose(self, A)
        A_tilda = (A - A_transpose)/2
        return A_tilda

    def asymmetryIndex(self, A):
        A_tilda = MatrixMath.matrixTilda(self, A)
        asymmetry = MatrixMath.norm(self, A_tilda) / MatrixMath.norm(self, A)
        return asymmetry

    def matrixRank(self, A):
        rank = np.linalg.matrix_rank(A)
        return rank

    # def hermitianTranspose(self, C):

    def complexVectSize(self, C):
        if MatrixMath.isComplex(self, C) != True:
            print("This is not a complex vector.")
            return
        size = np.linalg.norm(C)
        return size

    def matrixDataTyp(self, A):
        [rows, cols] = MatrixMath.size(self, A)
        typ0 = type(A[0][0])
        for row in range(rows):
            for col in range(cols):
                typ = type(A[row][col])
                if typ != typ0:
                    print("Error! More than one data type.")
                    return
        return typ

    def trace(self, A):
        """sum of the diagonal elements of a square matrix"""
        [rows, cols] = MatrixMath.size(self, A)
        if rows != cols:
            print("Matrix not square. Trace not possible")
            return
        trace = 0
        for row in range(rows):
            trace += A[row][row]
        tract = np.trace(A)
        return trace

    def isSymetric(self, A):
        [rows, cols] = MatrixMath.size(self, A)
        if rows != cols:
            print("Matrix not square, cannot be a symetric matrix")
            return
        symetric = True
        for row in range(rows):
            for col in range(cols):
                if A[row][col] != A[col][row]:
                    symetric = False
        return symetric

    """DISPLAY"""

    def displayVectors3D_1(self, v1=[0, 0, 0], v2=[0, 0, 0], v3=[0, 0, 0], v4=[0, 0, 0]):
        lenV1 = MatrixMath.vectorLength(self, v1)
        lenV2 = MatrixMath.vectorLength(self, v2)
        lenV3 = MatrixMath.vectorLength(self, v3)
        lenV4 = MatrixMath.vectorLength(self, v4)
        axis = int(max([lenV1, lenV2, lenV3]) * 1.5)

        # draw plane defined by V1 and V2
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))
        z1 = (-v3[0] * xx - v3[1] * yy) / v3[2]
        ax.plot([0, v1[0]], [0, v1[1]], [0, v1[2]], 'b')
        ax.plot([0, v2[0]], [0, v2[1]], [0, v2[2]], 'r')
        ax.plot([0, v3[0]], [0, v3[1]], [0, v3[2]], 'r')
        ax.plot_surface(xx, yy, z1)
        ax.view_init(azim=150, elev=45)
        plt.show()