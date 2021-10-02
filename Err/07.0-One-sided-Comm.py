import mpi4py.MPI as mpi
import numpy as np
import time

def main():
    rank = mpi.COMM_WORLD.Get_rank()
    n_proc = mpi.COMM_WORLD.Get_size()

    assert(n_proc == 2)

    buff = np.zeros(10, dtype='d')
    win = mpi.Win.Create(buff, 1, mpi.INFO_NULL, mpi.COMM_WORLD)

    if (rank == 0):
        win.Lock(1)
        win.Put([buff, mpi.DOUBLE], 1)
        win.Unlock(1)

        buff[-1] = 9
        time.sleep(5)

        win.Lock(1)
        win.Put([buff, mpi.DOUBLE], 1)
        win.Unlock(1)

    else:
        holder = np.zeros(10)
        failures = 0
        while (holder[-1] != 9):
            failures += 1
            win.Lock(0)
            win.Get([holder, mpi.DOUBLE], 0)
            win.Unlock(0)
        print('Took', failures, 'dials')
        print(f"holder = {holder}")

if __name__=='__main__':
    main()