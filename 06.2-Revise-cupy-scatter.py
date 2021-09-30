from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################
# Scatter
############################################################
if rank == 0:
    print("############################################################")
    print("# Scatter")
    print("############################################################")

arr = None

if rank == 0:
    arr = cp.arange(3*7*size).reshape(3*size,7)

    if arr.shape[0] % size == 0:
        arr = arr.reshape([size, int(arr.shape[0]/size), 7])

arr = comm.scatter(arr, root=0)

print(f"Rank #{rank}: arr = {arr}")




