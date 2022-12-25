import sys
import time
from random import Random
import numpy as np
from mpi4py import MPI
sys.path.append('./python')
sys.path.append('./apps')
from simple_training import train_cifar10, evaluate_cifar10
from models import ResNet9
import needle as ndl
import ddp

# Run this script: mpiexec -np NUM_GPU python train_resnet.py
# eg: mpiexec -np 3 python train_resnet.py


# class Partition(object):
#     """ Dataset-like object, but only access a subset of it. """
#
#     def __init__(self, data, index):
#         self.data = data
#         self.index = index
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]
#
#
# class DataPartitioner(object):
#     """ Partitions a dataset into different chuncks. """
#
#     def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
#         self.data = data
#         self.partitions = []
#         rng = Random()
#         rng.seed(seed)
#         data_len = len(data)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)
#
#         for frac in sizes:
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]
#
#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])
#
#
# def partition_dataset(dataset, batch_size, world_size, device, dtype):
#     bsz = batch_size//world_size
#     partition_sizes = [1.0/size for _ in range(world_size)]
#     partition = DataPartitioner(dataset, partition_sizes)
#     partition = partition.use(MPI.COMM_WORLD.Get_rank())
#     print(f'partitioned dataset length: {len(partition)}')
#     train_set = ndl.data.DataLoader(
#         dataset=partition, batch_size=bsz, shuffle=True, device=device, dtype=dtype)
#     return train_set, bsz
#
#
# def average_gradients(model, world_size):
#     for p in model.parameters():
#         if p.grad is None:
#             continue
#         sendbuf = np.ascontiguousarray(p.grad.numpy())
#         recvbuf = np.empty_like(sendbuf, dtype=p.dtype)
#         comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
#         recvbuf = recvbuf/world_size
#         p.grad.data = ndl.Tensor(recvbuf, device=device, dtype=p.grad.dtype)
#
#
# def broadcast_parameters(model, root_rank=0):
#     for p in model.parameters():
#         p_data = p.numpy()
#         p_data = comm.bcast(p_data, root=0)
#         p.data = ndl.Tensor(p_data, device=device, dtype=p.dtype)

if __name__ == "__main__":
    np.random.seed(0)
    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()
    #
    # device = ndl.cuda(rank)
    # print(f'Use cuda: {rank}')
    rank, device = ddp.init()

    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)

    train_set, bsz = ddp.partition_dataset(
        dataset, 128, device=device, dtype='float32')
    print(f'orignal dataset length: {len(dataset)}')

    model = ResNet9(device=device, dtype="float32")
    ddp.broadcast_parameters(model)

    model.train()
    correct, total_loss = 0, 0
    opt = ndl.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    opt = ddp.DistributedOptimizer(opt)
    loss_fn = ndl.nn.SoftmaxLoss()
    n_epochs = 1
    begin = time.time()
    for i in range(n_epochs):
        if rank == 0:
            print(f'epoch: {i+1}/{n_epochs}')
        count = 0
        for batch in train_set:
            opt.reset_grad()
            X, y = batch
            out = model(X)
            correct = np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            loss = loss_fn(out, y)
            loss.backward()
            # average_gradients(model, size)
            opt.step()
            acc = correct/(y.shape[0])
            if rank == 0 and count % 100 == 0:
                print(f'acc: {acc}; avg_loss: {loss.data.numpy()}')
            count += 1
    end = time.time()
    if rank == 0:
        print(f'Training Time: {end-begin}')
