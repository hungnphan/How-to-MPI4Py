from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###############################################################
# INITIALIZATION
###############################################################
# Data types
datatype_MPI = MPI.INT
datatype_np = np.int32

# Size of memory for RMA
n_element = 10
element_size = np.dtype(datatype_np).itemsize

# Arrays
local_arr = rank*10 + np.arange(5, dtype=datatype_np)       # 00 01 02 03 04
# print(f"Rank #{rank}: local_data = {local_arr}")

# Define neighbors
next_id = rank+1 if (rank != size-1) else 0
prev_id = rank-1 if (rank != 0) else size-1
# print(f"Rank #{rank}: next_id = {next_id}")

# Define RMA windows
win = MPI.Win.Allocate(n_element*element_size, element_size, MPI.INFO_NULL, comm)

# Define a local "pointer"
mem = np.frombuffer(win, dtype=datatype_np)


###############################################################
# ONE-SIDE COMM. WITH PUT
###############################################################
win.Lock(rank=next_id, lock_type=MPI.LOCK_SHARED, assertion=0)
win.Put(local_arr, next_id, [0,n_element,datatype_MPI])
win.Unlock(next_id)

comm.Barrier()

# Set some values to RMA windows
mem[:] = rank*11
arr = np.arange(5, dtype=datatype_np)*10 + rank
# print(f"Rank #{rank}: mem = {mem}"); sys.stdout.flush()

win.Lock(rank=prev_id, lock_type=MPI.LOCK_SHARED, assertion=0)
win.Put(arr, prev_id, [5,5,datatype_MPI])
win.Unlock(prev_id)

comm.Barrier()
# print(f"Rank #{rank}: mem = {mem}"); sys.stdout.flush()


###############################################################
# ONE-SIDE COMM. WITH GET
###############################################################

# Set some values to RMA windows
mem[:] = rank*10 + np.arange(n_element, dtype=datatype_np)
# print(f"Rank #{rank}: mem = {mem}"); sys.stdout.flush()

recv_buff = np.zeros(n_element,dtype=datatype_np)
# print(f"Rank #{rank}: recv_buff = {recv_buff}"); sys.stdout.flush()

win.Lock(rank=next_id, lock_type=MPI.LOCK_SHARED, assertion=0)
win.Get(recv_buff[rank:],next_id,[rank,5,datatype_MPI])
win.Unlock(next_id)

comm.Barrier()
print(f"Rank #{rank}: recv_buff = {recv_buff}"); sys.stdout.flush()















