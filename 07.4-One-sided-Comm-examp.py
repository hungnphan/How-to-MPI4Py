from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data types
datatype_MPI = MPI.INT
datatype_np  = dtlib.to_numpy_dtype(datatype_MPI)

# Size of memory for RMA
n_element = 10
element_size = datatype_np.itemsize     # int type = 4

# Arrays
local_arr = rank*10 + np.arange(5, dtype=datatype_np)
# print(f"Rank #{rank}: local_data = {local_arr}")

# Define neighbors
next_id = rank+1 if (rank != size-1) else 0
prev_id = rank-1 if (rank != 0) else size-1
# print(f"Rank #{rank}: next_id = {next_id}")

# Define RMA windows
win = MPI.Win.Allocate(n_element*element_size, element_size, MPI.INFO_NULL, comm)

# Define a local "pointer (!?)"
mem = np.frombuffer(win, dtype=datatype_np)

win.Lock(rank)
win.Put(local_arr, next_id, [0,n_element,datatype_MPI])
win.Unlock(rank)
# print(f"Rank #{rank}: mem = {mem}")

comm.Barrier()

mem[:] = rank*11
# print(f"Rank #{rank}: mem = {mem}")

arr = np.arange(5, dtype=datatype_np)*rank

win.Lock(rank)
win.Put(arr, prev_id, [0,5,datatype_MPI])
win.Unlock(rank)

print(f"Rank #{rank}: mem = {mem}")

