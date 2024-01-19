class EnsembleCommunicator(object):
    def __init__(self, global_comm, subsize, subx=True):
        size = global_comm.size
        if (size % subsize) != 0:
            msg = f"Size of subcommunicators {subsize} does not divide {size}"
            raise ValueError(msg)

        rank = global_comm.rank

        contiguous_comm = global_comm.Split(color=(rank//subsize),
                                            key=rank)

        noncontiguous_comm = global_comm.Split(color=(rank % subsize),
                                               key=rank)

        self.global_comm = global_comm

        self.xcomm = contiguous_comm if subx else noncontiguous_comm
        self.ycomm = noncontiguous_comm if subx else contiguous_comm
