from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################
# AllGather
############################################################
if rank == 0:
    print("############################################################")
    print("# AllGather")
    print("############################################################")
    sys.stdout.flush()

comm.barrier()

arr = cp.random.randint(20,size=[4])


arr = comm.allgather(arr) # list of torch tensor

arr = cp.stack(arr)

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")




