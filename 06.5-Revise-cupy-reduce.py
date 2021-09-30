from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

############################################################
# Reduce
############################################################
if rank == 0:
    print("############################################################")
    print("# Reduce")
    print("############################################################")
    sys.stdout.flush()

comm.barrier()

arr = cp.random.randint(20,size=[4])

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")
sys.stdout.flush()

comm.barrier()

arr = comm.reduce(arr, op=MPI.SUM, root=0) 

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")




