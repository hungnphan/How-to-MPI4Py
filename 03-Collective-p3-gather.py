from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = 5 * rank

data = comm.gather(data, root=0)

print(f"rank #{rank}: data = {data}")


