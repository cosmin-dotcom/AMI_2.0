from collections.abc import MutableMapping
from concurrent.futures import Future, Executor
from concurrent.futures import wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

import ami.abc.scheduler_factory
import ami.abc.worker_factory
from ami.abc import Index


@dataclass(slots=True)
class RunnerContextHelper:
    counter: int
    pool: ami.abc.worker_pool.WorkerExecutorInterface
    scheduler: ami.abc.scheduler.SchedulerInterface
    map: MutableMapping[Future, Index] = field(init=False, default_factory=dict)
    ranker_indices: Optional[Sequence[int]] = field(init=False, default=None)

    def schedule(self) -> Optional[Future]:


        if self.counter <= 0:
            # Exceeded maximum number of iterations.
            # Returns None, submits nothing.
            return None

        ranking_ok = self.ranker_indices is None

        # New data came out since last ranking.
        # Asks for an update.
        if self.scheduler.needs_new_ranking() and ranking_ok:
            self.ranker_indices, inp = self.scheduler.ranker_inputs()

            future = self.pool.submit_fit_and_rank(inp)
            self.map[future] = -1
            return future

        # Submits normal job
        idx = self.scheduler.next()
        inp = self.scheduler.parameters(idx)
        future = self.pool.submit_job(inp)
        self.map[future] = idx
        self.counter -= 1
        return future

    def report(self, future: Future) -> None:
        """Reports a result back directly from a future."""
        from ami.option import Some, Nothing
        index = self.map.pop(future)

        try:
            res = future.result()
            value = Some(res) if res is not None else Nothing
        except Exception:
            value = Nothing

        self.pool.release(future)

        if index >= 0:
            self.scheduler.set_result(index, value)

        if index == -1:
            assert self.ranker_indices is not None
            match value:
                case Some(sequence):
                    local_rank = np.asarray(sequence, dtype=int)
                    local_idx = np.asarray(self.ranker_indices, dtype=int)
                    ranked_idx = local_idx[local_rank]
                    self.scheduler.set_ranks(ranked_idx)
                case Nothing:
                    self.scheduler.set_ranks(Nothing)
            self.ranker_indices = None


@dataclass(slots=True, frozen=True)
class Runner:
    """Defines all moving parts for the program to execute.

    In essence, there are two kinds of types necessary for
    the full configuration: builders and normal instances.

    Factories are following the Factory/Builder pattern found
    in other languages: they are used to assemble all the parts
    together without forcing the use of a specific constructor
    signature nor providing partially-initialised objects.

    Parameters
    ----------

    scheduler: ami.abc.SchedulerInterface
        Scheduler.
    worker_pool: ami.abc.WorkerPoolInterface
        Worker pool.

    """
    from time import sleep
    scheduler: ami.abc.scheduler.SchedulerInterface
    worker_pool: ami.abc.worker_pool.WorkerPoolInterface

    def run(self, counter: int) -> None:
        n = len(self.worker_pool)
        with self.worker_pool as pool:
            ctx = RunnerContextHelper(counter, pool, self.scheduler)

            # Initialize the pool
            done = set()
            not_done = set(ctx.schedule() for _ in range(min(n, counter)))
            # Runs until count is reached.
            while len(not_done) > 0:
                done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
                for fut in done:
                    ctx.report(fut)
                    fut = ctx.schedule()
                    if fut is not None:
                        not_done.add(fut)

