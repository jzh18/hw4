from mpi4py import MPI
import needle as ndl
import random

def init(args):
    comm = args.comm
    size = comm.Get_size()
    rank = comm.Get_rank()
    device = ndl.cuda(rank)
    print(f'Use cuda: {rank}')

    if args.nccl:
        if rank==0:
            vec = device.get_id()
        else:
            vec = None
        vec = comm.bcast(vec, root=0)

        device.init_nccl(vec,rank,size)
    return rank, size, device

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset(dataset, batch_size, world_size, device, dtype):
    bsz = batch_size // world_size
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(MPI.COMM_WORLD.Get_rank())
    print(f'partitioned dataset length: {len(partition)}')
    train_set = ndl.data.DataLoader(
        dataset=partition, batch_size=bsz, shuffle=True, device=device, dtype=dtype)
    return train_set, bsz

def broadcast_parameters(model, args, rank=0):
    if args.nccl:
        for x in model.parameters():
            x.cached_data = x.realize_cached_data().broadcast(rank)
    else:
        comm = args.comm
        for p in model.parameters():
            p_data = p.numpy()
            p_data = comm.bcast(p_data, root=0)
            p.data = ndl.Tensor(p_data, device=p.device, dtype=p.dtype)
        
