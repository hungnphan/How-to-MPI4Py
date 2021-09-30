from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(10, dtype=np.int) * 10
else:
    data = None

data = comm.bcast(data, root=0)


print(f"Rank {rank} received {data}")