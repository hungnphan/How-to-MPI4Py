from mpi4py import MPI
import sys
import torch
import numpy as np
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################################
# AllReduce
############################################################
if rank == 0:
    print("############################################################")
    print("# AllReduce")
    print("############################################################")
    sys.stdout.flush()

comm.barrier()

arr = torch.randint(20,size=[4]).to(device)

arr = comm.allreduce(arr, op=MPI.SUM) 

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")




