from mpi4py import MPI
import numpy as np
from utils import print_in_order, print_once
from math import ceil

comm = MPI.COMM_WORLD
rank = comm.rank
nranks = comm.size

# initialise data

nx = 10
nt = 1

offset = 1 + nx*rank
data0 = offset + np.arange(nx, dtype=int)

# chunk sizes on each rank

chunk = ceil(nx/nranks)
nlast = max(nx - (nranks-1)*chunk, 0)

print_once(comm, f"chunk = {chunk}, nlast = {nlast}")

nlocal = chunk if rank != nranks-1 else nlast

# buffers

senddata = np.zeros_like(data0)
recvdata = np.zeros(nranks*nlocal, dtype=int)

senddata[:] = data0

# chunk counts/displacements

sendcounts = tuple([chunk for _ in range(nranks-1)] + [nlast])
senddispl = tuple((sum(sendcounts[:i]) for i in range(nranks)))

recvcounts = tuple((nlocal for _ in range(nranks)))
recvdispl = tuple((sum(recvcounts[:i]) for i in range(nranks)))

sendbuf = tuple((senddata, tuple((sendcounts, senddispl))))
recvbuf = tuple((recvdata, tuple((recvcounts, recvdispl))))

# do the comms

print_once(comm, "senddata:")
print_in_order(comm, f"rank {rank}: {senddata}")

comm.Alltoallv(sendbuf, recvbuf)
recvdata = recvdata.reshape((nranks, nlocal))

print_once(comm, "recvdata:")
print_in_order(comm, f"rank {rank}:\n{recvdata}")
