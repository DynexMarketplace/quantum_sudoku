# Demo: Quantum Sudoku
#
# Copyright (c) 2021-2024, Dynex Developers
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
#    of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be
#    used to endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import dimod
import dynex
from qubovert import boolean_var
import matplotlib.pyplot as plt 

dynex.test()

# Define Sudoku Puzzle Size (4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256)
# Sizes of greater than 25 will take some time to compile the Puzzle and Upload to Dynex
MainNet = False          # Either True or False
N = 4                    # Testnet maximum N = 4   |   # Mainnet can run all factors
Annealing = 500          # Recommended 500 for N = 4 - increase annealing when increasing N > 4

def generateSudokuPlot(matrix, values=True):
    n = matrix.shape[0]
    fig, ax = plt.subplots()
    
    if values:
        ax.matshow(matrix[tuple([n-i for i in range(1, n+1)]),:], cmap=plt.cm.Blues)
        min_val, max_val = 0, n
        for i in range(n):
            for j in range(n):
                ax.text(i, j, str(matrix[n-j-1,i]), va='center', ha='center')
        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(-0.5, n-0.5)
    else:
        ax.matshow(matrix, cmap=plt.cm.Blues)
    
    # Draw grid lines for mini subgrids
    sq_N = int(np.sqrt(n))
    for i in range(1, sq_N):
        ax.axhline(i * (n // sq_N) - 0.5, color='black', linewidth=2)
        ax.axvline(i * (n // sq_N) - 0.5, color='black', linewidth=2)
    
    plt.savefig('result.png')

def checkSudoku(grid):
    N = grid.shape[0]
    sq_N = int(np.sqrt(N))
    for i in range(N):
        if len(set(grid[i, :])) != N or len(set(grid[:, i])) != N:
            return False
    for i in range(0, N, sq_N):
        for j in range(0, N, sq_N):
            subgrid = grid[i:i+sq_N, j:j+sq_N]
            if len(set(subgrid.flatten())) != N:
                return False  
    return True

# Function to initialize the Sudoku puzzle
def solveSudoku(board, N):
    empty = findEmptyLocation(board, N)
    if not empty:
        return True
    row, col = empty
    for num in range(1, N+1):
        if isValid(board, row, col, num, N):
            board[row][col] = num
            if solveSudoku(board, N):
                return True
            board[row][col] = 0
    return False
    
def isValid(board, row, col, num, N):
    sq_N = int(np.sqrt(N))
    for x in range(N):
        if board[row][x] == num or board[x][col] == num:
            return False
    startRow = row - row % sq_N
    startCol = col - col % sq_N
    for i in range(sq_N):
        for j in range(sq_N):
            if board[i + startRow][j + startCol] == num:
                return False
    return True
    
def findEmptyLocation(board, N):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return (i, j)
    return None
    
def initializeSudoku(N):
    while True:
        num_bits = N**3
        sq_N = int(np.sqrt(N))
        z = np.array([i+1 for i in range(N)], dtype=np.int64)
        
        # Set initial values.
        idx_init = np.random.choice(np.arange(N*N), N, replace=False)
        a = np.array([[i for i in range(N)]+idx_init[j]*N for j in range(N)]).ravel()
        x_init = {a[i] : int(i%N == i//N) for i in range(N*N)}

        init_vec = np.zeros((num_bits, 1), dtype=np.int64)
        idx = np.array([k for (k, v) in x_init.items() if v])
        init_vec[idx] = 1

        sudoku_init = (np.kron(np.identity(N**2, dtype=np.int64), z) @ init_vec).reshape((N, N))
        
        # Check if the initial Sudoku is solvable
        board = sudoku_init.copy().astype(int)
        if solveSudoku(board, N):
            return sudoku_init, x_init

# Ensure N is a perfect square and at least 4
def isPerfectSquare(n):
    return int(np.sqrt(n))**2 == n

if not isPerfectSquare(N) or N < 4:
    raise ValueError("[DYNEX] N must be a perfect square number. (4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256)")

# Initialize the Sudoku Puzzle
print("[DYNEX] GENERATED SOLVABLE SUDOKU PUZZLE")
sudoku_init, x_init = initializeSudoku(N)

# Compute the Sudoku Puzzle
solution_found = False
attempts = 0

while not solution_found and attempts < 10:  # Add a limit to avoid infinite loops
    num_bits = N**3
    sq_N = int(np.sqrt(N))
    z = np.array([i+1 for i in range(N)], dtype=np.int64)

    # Initializing identity matrices and constructing constraints for rows and columns with high penalties
    penaltyRC = 10000 
    penaltySubGrid = 8000 
    penalty_value = 10000 
    idN = np.identity(N)
    idSqN = np.identity(sq_N)

    aRC = np.concatenate((np.kron(np.kron(idN, np.ones((1, N))), idN),  
                          np.kron(np.kron(np.ones((1, N)), idN), idN)))  
    aSubGrid = np.kron(np.kron(np.kron(idSqN, np.ones((1, sq_N))), 
                               np.kron(idSqN, np.ones((1, sq_N)))), idN)
    bRC = np.ones((2 * N * N, 1))
    bSubGrid = np.ones((N * N, 1))

    QRC = penaltyRC * (aRC.T @ aRC - 2 * np.diag(np.diag(aRC.T @ aRC)))
    QSubGrid = penaltySubGrid * (aSubGrid.T @ aSubGrid - 2 * np.diag(np.diag(aSubGrid.T @ aSubGrid)))

    qVal = np.zeros((num_bits, num_bits))
    for i in range(num_bits):
        block_row = i // N 
        for j in range(block_row * N, (block_row + 1) * N): 
            if i != j:
                qVal[i, j] = penalty_value 
    Q = QRC + QSubGrid + qVal

    # Creating BQM from the QUBO matrices
    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
    for i in range(num_bits):
        for j in range(i, num_bits):
            if i == j:
                bqm.add_variable(i, Q[i, i])
            else:
                bqm.add_interaction(i, j, Q[i, j])
    bqm.offset += penaltyRC * (bRC.T @ bRC)[0, 0]
    bqm.offset += penaltySubGrid * (bSubGrid.T @ bSubGrid)[0, 0]

    for i, val in x_init.items():
        bqm.fix_variable(i, val)

    # Compute on Dynex
    print("[DYNEX] UPLOADING JOB TO SAMPLER")
    model = dynex.BQM(bqm)
    sampler = dynex.DynexSampler(model, mainnet=MainNet, description='Quantum Sudoku Solution')
    sampleset = sampler.sample(num_reads=10000, annealing_time=Annealing, debugging=False, alpha=10, beta=1)
    solution = sampleset.first.sample
    print("\n", solution, "\n")
    
    # Convert solution into sudoku grid
    sol_vec = np.zeros((num_bits, 1), dtype=np.int64)
    for i in range(num_bits):
        if i in x_init:
            sol_vec[i] = x_init[i]
        else:
            sol_vec[i] = solution[i]
    sudoku_sol = (np.kron(np.identity(N**2, dtype=np.int64), z) @ sol_vec).reshape((N, N))

    if checkSudoku(sudoku_sol):
        solution_found = True
        print("[DYNEX] THE SOLUTION IS VALID\n")
        generateSudokuPlot(sudoku_sol)
    else:
        print("[DYNEX] THE SOLUTION IS INVALID, RETRYING QUBO")
    
    attempts += 1

if not solution_found:
    print("[DYNEX] THE SOLUTION NOT FOUND IN 10 ATTEMPTS. PLEASE INCREASE ANNEALING")



