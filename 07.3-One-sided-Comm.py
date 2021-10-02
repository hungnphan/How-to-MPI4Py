from mpi4py import MPI
from mpi4py.util import dtlib
import numpy as np
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data types
datatype_MPI = MPI.INT
datatype_np  = dtlib.to_numpy_dtype(datatype_MPI)

if rank == 0:
    print(f"datatype_np = {datatype_np}")

# Size of memory for RMA
n_element = 10
element_size = datatype_np.itemsize

# Create windows
# [ALT-01]
win = MPI.Win.Allocate(n_element*element_size, element_size, MPI.INFO_NULL, comm)
# [ALT-02]
# buff = np.zeros([n_element], dtype=datatype_np)
# win = MPI.Win.Create(memory=buff, disp_unit=1, info=MPI.INFO_NULL, comm=comm)

if rank == 0:

    arr = np.random.randint(0,10,n_element,dtype=datatype_np)
    print(f"random array on rank #{rank}: {arr}")

    win.Lock(rank=1)
    win.Put([arr, datatype_MPI], 1)
    win.Unlock(rank=1)

else: # rank == 1
    time.sleep(15)

    buff = np.zeros(n_element, dtype=datatype_np)

    # win.Lock(rank=0)
    # win.Get([buff, datatype_MPI], 0)
    # win.Unlock(rank=0)
    # print(f"buff on rank 1: {buff}")

    mem = np.frombuffer(win, datatype_np)
    print(f"mem on rank 1: {mem}")
