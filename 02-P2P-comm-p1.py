from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"This process is rank #{rank}")

comm.Barrier()

if rank == 0:
    data = {'a': 7, 'b': 3.14}
else:
    data = comm.recv(source=0,tag=11)

comm.Barrier()
print(f"Rank #{rank}: data = {data}")

