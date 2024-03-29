import abc
import numpy as np
from mpi4py import MPI

__all__ = ["Transposer", "Transposerv", "Transposerw", "PaddedTransposer"]


class TransposerInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        pass

    @abc.abstractmethod
    def forward(self, x, y):
        pass

    @abc.abstractmethod
    def backward(self, y, x):
        pass


class TransposerBase(TransposerInterface):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        if type(xpart) is int:
            xpart = tuple(xpart for _ in range(comm.size))
        if type(ypart) is int:
            ypart = tuple(ypart for _ in range(comm.size))

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        self._xshape = tuple((nylocal*self.nx,))
        self._yshape = tuple((nxlocal*self.ny,))

        self._xbuf = np.zeros(self._xshape, dtype=dtype)
        self._ybuf = np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xbuf.dtype

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        self.Alltoall(self._xmsg, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        self.Alltoall(self._ymsg, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _get_buf(self, msg):
        return NotImplementedError

    def _packx(self, x, xm):
        xbuf = self._get_buf(xm)
        self._pack(x, xbuf)

    def _packy(self, y, ym):
        ybuf = self._get_buf(ym)
        self._pack(y, ybuf)

    def _unpackx(self, xm, x):
        xbuf = self._get_buf(xm)
        self._unpack(xbuf, x, self.ypart, self.xpart)

    def _unpacky(self, ym, y):
        ybuf = self._get_buf(ym)
        self._unpack(ybuf, y, self.xpart, self.ypart)

    def _pack(self, src, dst):
        dst.reshape(src.T.shape)[:] = src.T

    def _unpack(self, src, dst, srcpart, dstpart):
        rank = self.comm.rank
        for r in range(self.comm.size):
            lsrc = srcpart[rank]
            ldst = dstpart[r]
            b = sum(dstpart[:r])
            e = sum(dstpart[:r+1])
            dstbuf = src[lsrc*b:lsrc*e].reshape((lsrc, ldst))
            dst[:, b:e] = dstbuf

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype


class Transposer(TransposerBase):
    """
    Class for transposing a 2D distributed array.

    The distribution:
        - must be one dimensional i.e. each rank only holds either
          entire rows or entire columns at any one time.
        - must be uniform i.e. all ranks hold the same number of
          either rows or columns at any one time.

    Provides methods for the forward and backward transpose.
    Forward transpose assumes each rank hold entire rows (x coord) and
    transposes so each rank holds entire columns (y coord).
    Backward transpose does the reverse operation.

    The uniform distribution means that MPI_Alltoall can be used
    for the communication round.
    """

    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        """
        :arg xpart: The x partition i.e. the number of columns on each rank
            after a backward transpose. Can be either a tuple or int. If
            xpart is a tuple then it must contain exactly comm.size
            identical elements.
        :arg ypart: The y partition i.e. the number of rows on each rank
            after a forward transpose. Can be either a tuple or int. If
            ypart is a tuple then it must contain exactly comm.size
            identical elements.
        :arg comm: The MPI communicator that the array is distributed over.
        :arg dtype: the type of the array elements. Must be a valid numpy
            array element type.
        """
        super().__init__(xpart, ypart, comm=comm, dtype=dtype)

        assert len(set(xpart)) == 1
        assert len(set(ypart)) == 1

        self._xmsg = self._xbuf
        self._ymsg = self._ybuf

        self.Alltoall = self.comm.Alltoall

    def _get_buf(self, msg):
        return msg


class Transposerv(TransposerBase):
    """
    Class for transposing a 2D distributed array.

    The distribution:
        - must be one dimensional i.e. each rank only holds either
          entire rows or entire columns at any one time.
        - may be non-uniform i.e. each rank can hold a different number
          of either rows or columns at any one time.

    Provides methods for the forward and backward transpose.
    Forward transpose assumes each rank hold entire rows (x coord) and
    transposes so each rank holds entire columns (y coord).
    Backward transpose does the reverse operation.

    The non-uniform distribution means that MPI_Alltoallv must be used
    for the communication round.
    """

    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        """
        :arg xpart: The x partition i.e. the number of columns on each rank
            after a backward transpose. Can be either a tuple or int. If
            xpart is a tuple then it must contain exactly comm.size elements.
        :arg ypart: The y partition i.e. the number of rows on each rank
            after a forward transpose. Can be either a tuple or int. If
            ypart is a tuple then it must contain exactly comm.size elements.
        :arg comm: The MPI communicator that the array is distributed over.
        :arg dtype: the type of the array elements. Must be a valid numpy
            array element type.
        """
        super().__init__(xpart, ypart, comm=comm, dtype=dtype)

        rank = comm.rank
        nranks = comm.size

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self._xcounts = tuple((nylocal*nxremote for nxremote in xpart))
        self._ycounts = tuple((nyremote*nxlocal for nyremote in ypart))

        self._xdispl = tuple((sum(self._xcounts[:r]) for r in range(nranks)))
        self._ydispl = tuple((sum(self._ycounts[:r]) for r in range(nranks)))

        self._xmsg = tuple((self._xbuf, tuple((self._xcounts, self._xdispl))))
        self._ymsg = tuple((self._ybuf, tuple((self._ycounts, self._ydispl))))

        self.Alltoall = self.comm.Alltoallv

    def _get_buf(self, msg):
        return msg[0]


class Transposerw(TransposerInterface):
    """
    Class for transposing a 2D distributed array.

    The distribution:
        - must be one dimensional i.e. each rank only holds either
          entire rows or entire columns at any one time.
        - must be uniform i.e. all ranks hold the same number of
          either rows or columns at any one time.

    Provides methods for the forward and backward transpose.
    Forward transpose assumes each rank hold entire rows (x coord) and
    transposes so each rank holds entire columns (y coord).
    Backward transpose does the reverse operation.

    The transposes use the implementation from the mpi4py-fft
    library with MPI_Alltoallw.
    """
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        """
        :arg xpart: The x partition i.e. the number of columns on each rank
            after a backward transpose. Can be either a tuple or int. If
            xpart is a tuple then it must contain exactly comm.size
            identical elements.
        :arg ypart: The y partition i.e. the number of rows on each rank
            after a forward transpose. Can be either a tuple or int. If
            ypart is a tuple then it must contain exactly comm.size
            identical elements.
        :arg comm: The MPI communicator that the array is distributed over.
        :arg dtype: the type of the array elements. Must be a valid numpy
            array element type.
        """
        from mpitranspose.pencil import Pencil, Subcomm

        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        subcomm = Subcomm(comm, [0, 1])
        sizes = np.array([self.ny, self.nx], dtype=int)
        p0 = Pencil(subcomm, sizes, axis=1)
        p1 = p0.pencil(0)

        self.xshape = p0.subshape
        self.yshape = tuple(reversed(p1.subshape))

        self._y = np.zeros(p1.subshape, dtype=dtype)

        assert self.xshape[0] == ypart[rank]
        assert self.xshape[1] == self.nx

        assert self.yshape[0] == xpart[rank]
        assert self.yshape[1] == self.ny

        self.transfer = p0.transfer(p1, dtype)
        self.dtype = self.transfer.dtype

    def forward(self, x, y):
        self.transfer.forward(x, self._y)
        y.T[:] = self._y

    def backward(self, y, x):
        self._y[:] = y.T
        self.transfer.backward(self._y, x)


class PaddedTransposer(TransposerInterface):
    """
    Class for transposing a 2D distributed array.

    The distribution:
        - must be one dimensional i.e. each rank only holds either
          entire rows or entire columns at any one time.
        - may be non-uniform in x i.e. each rank can hold a different
          number of columns.
        - must be uniform in y i.e. all ranks hold the same number of
          rows.

    Provides methods for the forward and backward transpose.
    Forward transpose assumes each rank hold entire rows (x coord) and
    transposes so each rank holds entire columns (y coord).
    Backward transpose does the reverse operation.

    This implementation assumes that the x partition is nearly uniform
    i.e. the largest number of columns on any rank is only slightly
    larger than the number of columns on any other rank (relatively).
    For the transpose, the local arrays are zero-padded so they are
    the same size on every rank. This means that the communications
    can be implemented with MPI_Alltoall, which can be more easily
    optimised than MPI_Alltoallv or MPI_Alltoallw.

    This assumes that the rows are much longer than the columns
    (sum(xpart) >> sum(ypart)) and that the benefit of using more
    optimised collective operations is greater than the overhead
    of the additional communication volume.
    """

    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        """
        :arg xpart: The x partition i.e. the number of columns on each rank
            after a backward transpose. Can be either a tuple or int. If
            xpart is a tuple then it must contain exactly comm.size elements.
        :arg ypart: The y partition i.e. the number of rows on each rank
            after a forward transpose. Can be either a tuple or int. If
            ypart is a tuple then it must contain exactly comm.size
            identical elements.
        :arg comm: The MPI communicator that the array is distributed over.
        :arg dtype: the type of the array elements. Must be a valid numpy
            array element type.
        """
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        assert len(set(ypart)) == 1

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        nxlocal_pad = max(xpart)
        nxpad = comm.size*nxlocal_pad
        msglen = nxlocal_pad*nylocal

        self._xshape = tuple((nylocal*nxpad,))
        self._yshape = tuple((nxlocal_pad*self.ny,))

        self._xpadshape = tuple((nxpad, nylocal))
        self._ypadshape = tuple((comm.size, msglen))

        self._xslice = np.s_[:, :self.nx]
        self._yslice = np.s_[:nxlocal, :]

        self._xbuf = np.zeros(self._xshape, dtype=dtype)
        self._ybuf = np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xbuf.dtype

        self._xmsg = self._xbuf
        self._ymsg = self._ybuf

        self.Alltoall = self.comm.Alltoall

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        self.Alltoall(self._xmsg, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        self.Alltoall(self._ymsg, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _packx(self, x, xm):
        nx = self.nx
        xmsg = xm.reshape(self._xpadshape).T
        for r in range(self.comm.size):
            xb = sum(self.xpart[:r])
            mlen = self.xpart[r]
            xpad = max(self.xpart)
            yb = r*xpad
            xmsg[:nx, yb:yb+mlen] = x[:, xb:xb+mlen]

    def _unpackx(self, xm, x):
        nx = self.nx
        xmsg = xm.reshape(self._xpadshape).T
        for r in range(self.comm.size):
            xb = sum(self.xpart[:r])
            mlen = self.xpart[r]
            xpad = max(self.xpart)
            yb = r*xpad
            x[:, xb:xb+mlen] = xmsg[:nx, yb:yb+mlen]

    def _packy(self, y, ym):
        rank = self.comm.rank
        xpad = max(self.xpart)
        nylocal = self.ypart[rank]
        nxlocal = self.xpart[rank]
        ybuf = ym.reshape(self._ypadshape)

        for r in range(self.comm.size):
            ymsg = ybuf[r, :].reshape(xpad, nylocal)
            yb = sum(self.ypart[:r])
            yl = nylocal
            ymsg[:nxlocal, :] = y[:, yb:yb+yl]

    def _unpacky(self, ym, y):
        rank = self.comm.rank
        xpad = max(self.xpart)
        nylocal = self.ypart[rank]
        nxlocal = self.xpart[rank]
        ybuf = ym.reshape(self._ypadshape)

        for r in range(self.comm.size):
            ymsg = ybuf[r, :].reshape(xpad, nylocal)
            yb = sum(self.ypart[:r])
            yl = nylocal
            y[:, yb:yb+yl] = ymsg[:nxlocal, :]

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
