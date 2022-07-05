import wandb
import csv
import os
import torch


class OnlineWandbLogger(object):

    def __init__(self,
                 project,
                 entity,
                 name,
                 offline=True,
                 config=None):

        super().__init__()

        self.project = project
        self.entity = entity
        self.name = name
        self.offline = offline

        if self.offline:
            os.environ['WANDB_MODE'] = 'offline'

        self.run = wandb.init(project=project,
                              entity=entity,
                              name=name,
                              config=config)



        self.sequence = None

    def set_sequence(self, sequence):
        self.sequence = sequence

    def log(self, results_dict, step=None):
        mapped_dict = {os.path.join(str(self.sequence), k): v for k, v in results_dict.items()}
        self.run.log(mapped_dict, step=step)


class OnlineCSVLogger(object):

    def __init__(self,
                 save_dir,
                 version='logs'):

        super().__init__()

        self.save_dir = save_dir
        self.version = version

        os.mkdir(os.path.join(self.save_dir, self.version))

        self.metrics = []

        self.metrics_file_path = os.path.join(self.save_dir, self.version)
        self.sequence = None

    def set_sequence(self, sequence):
        self.sequence = sequence

    def log(self, results_dict, step=None):
        self.log_metrics(results_dict, step)
        self.save()

    def log_metrics(self, metrics_dict, step=None) -> None:
        """Record metrics"""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded hparams and metrics into files"""

        if not self.metrics:
            return

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        log_path = os.path.join(self.metrics_file_path, str(self.sequence)+'.csv')

        with open(log_path, "w", newline="") as f:
            self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
            self.writer.writeheader()
            self.writer.writerows(self.metrics)


