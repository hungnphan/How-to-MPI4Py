from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(size) ** 2
else:
    data = None

data = comm.scatter(data, root=0)

# assert data == (rank+1)**2


print(f"Rank {rank} received {data}")