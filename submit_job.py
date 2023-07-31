from dataclasses import dataclass

from train import add_arguments, train
import sys
from submitit.helpers import DelayedSubmission


class Trainer: 
    def run(self, args): 
        self.args = args 
        train(args)
        
    def checkpoint(self):
        return DelayedSubmission(self.run, self.args)


if __name__ == "__main__":
    import submitit

    executor = submitit.AutoExecutor(folder=".submitit")
    executor.update_parameters(
        mem_gb=8,
        cpus_per_task=16,
        timeout_min=180,
        gres="gpu:1",
        slurm_partition="a40,t4v2,rtx6000",
    )
    
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    
    job = executor.submit(train, args.train)
    print(f"Submitted job: {job.job_id}")
    print(f"Job stdout at: {job.paths.stdout}")
    print(f"Job stderr at: {job.paths.stderr}")
