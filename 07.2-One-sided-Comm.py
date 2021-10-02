from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import cupy as cp
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

datatype = MPI.INT
# np_dtype = dtlib.to_numpy_dtype(datatype)

itemsize = datatype.Get_size()
N = 10
win_size = N * itemsize
win = MPI.Win.Allocate(win_size, comm=comm)


if rank == 0:

    x = np.arange(10).astype(np.int)

    send_rank = 1

    win.Lock(rank=send_rank)
    win.Put([x, datatype], target_rank=send_rank)         # local buf -> win on rank 0
    win.Unlock(rank=send_rank)


else:


    y = np.zeros(N, dtype=np.int)

    print(f"y on rank 1: {y}")

    recv_rank = 1
    win.Lock(rank=recv_rank)
    win.Get([y, datatype], target_rank=recv_rank)         # on rank {rank}: win on rank 0 -> buf
    win.Unlock(rank=recv_rank)

    print(f"y on rank 1: {y}")

    mem = np.frombuffer(win, dtype=dtlib.to_numpy_dtype(datatype))

    print(f"mem on rank 1: {mem}")

# PUT: local buf -> remote win (target)
# GET: remote win (target) -> local buf