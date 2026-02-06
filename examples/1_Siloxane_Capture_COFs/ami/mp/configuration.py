from dataclasses import dataclass

import ami
import ami.abc.scheduler_factory
import ami.abc.worker_factory
import ami.mp.runner


@dataclass(slots=True, frozen=True)
class Configuration:
    """Defines all moving parts for the program to execute.

    In essence, there are two kinds of types necessary for
    the full configuration: builders and normal instances.

    Factories are following the Factory/Builder pattern found
    in other languages: they are used to assemble all the parts
    together without forcing the use of a specific constructor
    signature nor providing partially-initialised objects.

    Parameters
    ----------

    scheduler: ami.abc.SchedulerFactoryInterface
        Scheduler factory.
    worker: ami.abc.WorkerFactoryInterface
        Worker factory.
    pool: ami.abc.WorkerPoolFactoryInterface
        WorkerPool factory.
    truth: ami.abc.TruthInterface
        Truth source: a calculator executed by a worker.
    data: ami.abc.DataManagerInterface
        Storage backend.
    ranker: ami.abc.RankerInterface
        Acquisition function used to transform data to rank.


    """
    # Builders/Factories
    scheduler: ami.abc.scheduler_factory.SchedulerFactoryInterface
    worker: ami.abc.worker_factory.WorkerFactoryInterface
    pool: ami.abc.worker_pool.WorkerPoolFactoryInterface

    # Direct instances
    data: ami.abc.data_manager.DataManagerInterface
    truth: ami.abc.calculator.CalculatorInterface
    initial_ranker: ami.abc.ranker.RankerInterface
    ranker: ami.abc.ranker.RankerInterface

    def build(self) -> ami.mp.runner.Runner:
        worker_pool = self._configure_worker_pool()
        scheduler = self._build_scheduler(worker_pool)
        return ami.mp.runner.Runner(scheduler=scheduler, worker_pool=worker_pool)

    def _build_scheduler(self, worker_pool: ami.abc.WorkerPoolInterface) -> ami.abc.SchedulerInterface:
        scheduler_builder = self.scheduler
        initial_ranker = self.initial_ranker
        scheduler_builder.set_ranker_schema(initial_ranker.schema())
        scheduler_builder.set_truth_schema(self.truth.schema())
        scheduler_builder.set_data_manager(self.data)
        scheduler_builder.set_worker_pool(worker_pool)
        scheduler_builder.set_initial_ranker(initial_ranker)
        return scheduler_builder.build().unwrap()

    def _configure_worker_pool(self) -> ami.abc.worker_pool.WorkerPoolInterface:
        pool = self.pool
        worker_factory = self.worker
        worker_factory.set_truth(self.truth)
        worker_factory.set_ranker(self.ranker)
        pool.set_worker_factory(worker_factory)
        return pool.build().unwrap()
