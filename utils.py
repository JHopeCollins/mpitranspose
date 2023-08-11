
def print_in_order(comm, string):
    comm.Barrier()
    for i in range(comm.size):
        if comm.rank == i:
            print(string)
        comm.Barrier()
    comm.Barrier()
