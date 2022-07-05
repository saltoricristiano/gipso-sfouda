import torch
import pytorch_lightning as plt
import MinkowskiEngine as ME
import open3d as o3d


class BasePipeline(object):

    def __init__(self,
                 model=None,
                 loss=None,
                 optimizer=None,
                 scheduler=None):

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        raise NotImplementedError

    def single_gpu_train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

