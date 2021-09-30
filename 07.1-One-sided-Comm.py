from mpi4py import MPI
# from mpi4py.util import dtlib
import numpy as np
import cupy as cp
import sys


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

datatype = MPI.FLOAT
# np_dtype = dtlib.to_numpy_dtype(datatype)
itemsize = datatype.Get_size()

N = 10

win_size = N * itemsize if rank == 0 else 0

win = MPI.Win.Allocate(win_size, comm=comm)
buf = np.empty(N, dtype=np.float)

if rank == 0:
    buf.fill(rank)
    print(f"buf = {buf}")

    win.Lock(rank=0)
    win.Put(buf, target_rank=0)         # local buf -> win on rank 0
    win.Unlock(rank=0)

    comm.Barrier()

else:
    comm.Barrier()

    win.Lock(rank=0)
    win.Get(buf, target_rank=0)         # on rank {rank}: win on rank 0 -> buf
    win.Unlock(rank=0)

print(f"rank #{rank}: data={buf}")

# PUT: local buf -> remote win (target)
# GET: remote win (target) -> local buf