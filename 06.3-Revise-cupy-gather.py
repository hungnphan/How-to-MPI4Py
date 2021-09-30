from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################
# Gather
############################################################
if rank == 0:
    print("############################################################")
    print("# Gather")
    print("############################################################")

arr = cp.random.randint(20,size=[4])


arr = comm.gather(arr, root=0) # list of torch tensor

if rank==0:
    arr = cp.stack(arr)

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")




