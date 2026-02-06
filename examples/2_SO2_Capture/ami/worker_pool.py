from concurrent.futures import ProcessPoolExecutor, Future, Executor
from contextlib import ExitStack
from dataclasses import dataclass, field, fields
from queue import Queue
from typing import Set, MutableMapping, Optional, Sequence

import ami.abc
from ami.abc import WorkerFactoryInterface
from ami.abc import WorkerInterface, WorkerExecutorInterface
from ami.factory import DataclassFactory
from ami.serialized_opaque import SerializedOpaque
from ami.surrogate_input import SurrogateInput

Index = int


def fit_and_rank(worker, inp: SurrogateInput) -> Optional[Sequence[Index]]:
    worker.fit(inp.known_x, inp.known_y)
    return worker.rank(inp.unknown_x)


@dataclass(frozen=True, slots=True)
class SharedMemoryExecutor(WorkerExecutorInterface):
    pool: Executor
    idle: Queue[WorkerInterface]
    busy: MutableMapping[Future, WorkerInterface] = field(default_factory=dict)

    def submit_fit_and_rank(self, inp: SurrogateInput) -> Future:

        w = self.idle.get()
        future = self.pool.submit(fit_and_rank, w, inp)
        self.busy[future] = w
        return future

    def submit_job(self, inp: SerializedOpaque) -> Future:
        w = self.idle.get()

        future = self.pool.submit(w.calculate, inp)
        self.busy[future] = w
        return future

    def release(self, future: Future):
        q = self.busy[future]
        self.idle.put(q)


@dataclass(slots=True, frozen=True)
class SingleNodeWorkerPool(ami.abc.WorkerPoolInterface):
    ncpus: int
    worker_factory: ami.abc.WorkerFactoryInterface
    stack: ExitStack = field(default_factory=ExitStack, init=False)

    def __enter__(self) -> SharedMemoryExecutor:
        pool = self.stack.enter_context(ProcessPoolExecutor(max_workers=self.ncpus))
        # # Could implement CPU pinning with affinity
        # affinity = self.cpu_affinity()
        # work_numbers = sorted(affinity)[:self.ncpus]
        q = Queue()
        for _ in range(self.ncpus):
            q.put(self.worker_factory.build().unwrap())
        return SharedMemoryExecutor(pool, idle=q)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.close()

    @staticmethod
    def cpu_affinity() -> Set[int]:
        from os import sched_getaffinity
        return sched_getaffinity(0)

    def __len__(self):
        return self.ncpus


@dataclass(frozen=True, slots=True)
class SingleNodeWorkerPoolFactory(DataclassFactory, ami.abc.WorkerPoolFactoryInterface):
    dataclass = SingleNodeWorkerPool

    def set_worker_factory(self, worker_factory: WorkerFactoryInterface) -> None:
        self.set("worker_factory", worker_factory)


