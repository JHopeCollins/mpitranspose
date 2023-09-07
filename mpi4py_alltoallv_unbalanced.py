from mpi4py import MPI
import numpy as np
from utils import print_in_order, print_once
from math import ceil
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.rank
nranks = comm.size

# initialise data

nx = 5
part = tuple((2, 1, 3))
nt = sum(part)

assert len(part) == nranks

data0 = np.zeros((part[rank], nx), dtype=int)

for i in range(part[rank]):
    offset = 1 + nx*(sum(part[:rank]) + i)
    data0[i, :] = offset + np.arange(nx, dtype=int)

print_once(comm, "data:")
print_in_order(comm, data0)
sleep(0.1)
comm.Barrier()

# chunk sizes on each rank

chunk = ceil(nx/nranks)
nlast = max(nx - (nranks-1)*chunk, 0)

nlocal = chunk if rank != nranks-1 else nlast

# buffers

senddata = np.zeros(part[rank]*nx, dtype=int)
recvdata = np.zeros((sum(part), nlocal), dtype=int)

off = 0
for i in range(nranks):
    for j in range(part[rank]):
        length = chunk if (i != nranks-1) else nlast
        begin = i*chunk
        end = begin + length
        senddata[off:off+length] = data0[j, begin:end]
        off += length

comm.Barrier()
print_once(comm, "senddata:")
print_in_order(comm, f"{senddata}")

# chunk counts/displacements

sendcounts = tuple([part[rank]*chunk for i in range(nranks-1)]
                   + [part[rank]*nlast])
senddispl = tuple((sum(sendcounts[:i]) for i in range(nranks)))

recvcounts = tuple((part[i]*nlocal for i in range(nranks)))
recvdispl = tuple((sum(recvcounts[:i]) for i in range(nranks)))

sendbuf = tuple((senddata, tuple((sendcounts, senddispl))))
recvbuf = tuple((recvdata, tuple((recvcounts, recvdispl))))

# do the comms

comm.Alltoallv(sendbuf, recvbuf)
recvdata = recvdata.T

print_once(comm, "recvdata:")
print_in_order(comm, f"{recvdata}")
