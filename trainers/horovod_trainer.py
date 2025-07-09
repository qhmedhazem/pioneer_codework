import os
import torch
from trainers.base_trainer import BaseTrainer, sample_to_cuda
from utils.config import prep_logger_and_checkpoint
from utils.logging import print_config
from utils.logging import AvgMeter


class HorovodTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True

        self.avg_loss = AvgMeter(50)
        self.dtype = kwargs.get("dtype", None)

    @property
    def proc_rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def fit(self, module):
        module.trainer = self
        prep_logger_and_checkpoint(module)
        print_config(module.config)

        module = module.to('cuda')
        module.configure_optimizers()

        optimizer = module.optimizer
        scheduler = module.scheduler

        train_dataloader = module.train_dataloader()
        val_dataloaders = module.val_dataloader()

        if self.validate_first:
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)

        for epoch in range(module.current_epoch, self.max_epochs):
            self.train(train_dataloader, module, optimizer)
            validation_output = self.validate(val_dataloaders, module)
            self.check_and_save(module, validation_output)
            module.current_epoch += 1
            scheduler.step()

    def train(self, dataloader, module, optimizer):
        module.train()
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(module.current_epoch)

        progress_bar = self.train_progress_bar(
            dataloader, module.config.datasets.train)
        outputs = []
        for i, batch in progress_bar:
            optimizer.zero_grad()
            batch = sample_to_cuda(batch)
            output = module.training_step(batch, i)
            output['loss'].backward()
            optimizer.step()
            output['loss'] = output['loss'].detach()
            outputs.append(output)
            if self.is_rank_0:
                progress_bar.set_description(
                    f'Epoch {module.current_epoch} | Avg.Loss {self.avg_loss(output["loss"].item()):.4f}')
        return module.training_epoch_end(outputs)

    def validate(self, dataloaders, module):
        module.eval()
        all_outputs = []
        for n, dataloader in enumerate(dataloaders):
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.validation, n)
            outputs = []
            for i, batch in progress_bar:
                batch = sample_to_cuda(batch)
                output = module.validation_step(batch, i, n)
                outputs.append(output)
            all_outputs.append(outputs)
        return module.validation_epoch_end(all_outputs)

    def test(self, module):
        module = module.to('cuda', dtype=self.dtype)
        test_dataloaders = module.test_dataloader()
        self.evaluate(test_dataloaders, module)

    @torch.no_grad()
    def evaluate(self, dataloaders, module):
        module.eval()
        all_outputs = []
        for n, dataloader in enumerate(dataloaders):
            progress_bar = self.val_progress_bar(
                dataloader, module.config.datasets.test, n)
            outputs = []
            for i, batch in progress_bar:
                batch = sample_to_cuda(batch, self.dtype)
                output = module.test_step(batch, i, n)
                outputs.append(output)
            all_outputs.append(outputs)
        return module.test_epoch_end(all_outputs)
