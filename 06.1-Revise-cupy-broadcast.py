from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################
# Broadcast
############################################################
if rank == 0:
    print("############################################################")
    print("# Broadcast")
    print("############################################################")

if rank == 0:
    arr = cp.random.randint(20, size=[8])
else:
    arr = None

arr = comm.bcast(arr, root=0)
print(f"Rank #{rank}: arr = {arr}")

