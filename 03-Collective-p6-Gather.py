from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = np.zeros(7, dtype='i') + rank
recvbuf = None

if rank == 0:
    recvbuf = np.empty([size, 7], dtype='i')

comm.Gather(sendbuf, recvbuf, root=0)

print(f"Rank #{rank}: data={recvbuf}")
