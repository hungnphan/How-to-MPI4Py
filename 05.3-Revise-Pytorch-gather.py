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
# Gather
############################################################
if rank == 0:
    print("############################################################")
    print("# Gather")
    print("############################################################")

arr = torch.randint(20,size=[4]).to(device)


arr = comm.gather(arr, root=0) # list of torch tensor

if rank==0:
    arr = torch.stack(arr)

print(f"Rank #{rank}: arr = {arr}, type(arr) = {type(arr)}")




