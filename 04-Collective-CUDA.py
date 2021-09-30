from mpi4py import MPI
import cupy as cp


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    data = cp.random.randint(256,size=[64,64,1024],dtype=cp.uint8)
else:
    data = None

data = comm.bcast(data, root=0)


print(f"Rank {rank} received data, type {type(data)}")
