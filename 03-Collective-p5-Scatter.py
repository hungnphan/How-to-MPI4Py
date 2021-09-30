from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None

if rank == 0:
    sendbuf = np.empty([size, 7], dtype='i')
    sendbuf.T[...,:] = range(size)

recvbuf = np.empty(7, dtype='i')

comm.Scatter(sendbuf, recvbuf, root=0)


print(f"Rank #{rank}: data = {recvbuf}")
