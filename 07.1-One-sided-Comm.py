import mpi4py.MPI as mpi
import numpy as np
import time

def main():
    rank = mpi.COMM_WORLD.Get_rank()
    n_proc = mpi.COMM_WORLD.Get_size()

    buff = np.zeros(10, dtype='d')
    win = mpi.Win.Create(buff, 1, mpi.INFO_NULL, mpi.COMM_WORLD)

    # itemsize = mpi.DOUBLE.Get_size()
    # n_item = 10
    # buff = mpi.Alloc_mem(n_item*itemsize, info=mpi.INFO_NULL)
    # print(type(buff))
    # win = mpi.Win.Create(buff, 1, mpi.INFO_NULL, mpi.COMM_WORLD)

    if (rank == 0):
        buff.fill(1)


        win.Lock(1)
        win.Put([buff, mpi.DOUBLE], 1)
        win.Unlock(1)

        buff[-1] = 9
        time.sleep(5)

        arr = np.arange(10)

        win.Lock(1)
        win.Put([arr, mpi.DOUBLE], 1)
        win.Unlock(1)

    else:
        holder = np.zeros(10)
        failures = 0


        time.sleep(15)

        # while (holder[-1] != 9):
            # failures += 1
        win.Lock(1)
        win.Get([holder, mpi.DOUBLE], 1)
        win.Unlock(1)

            # print(f"holder = {holder}")
            # holder = win.tomemory()

        # print('Took', failures, 'dials')
        print(f"holder = {holder}")

if __name__=='__main__':
    main()

