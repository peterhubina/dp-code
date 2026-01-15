from project.base.logging import Logger
from pathlib import Path
import torch

#------------------------------------------------------------------------------
#   ModelCheckpoint 
#------------------------------------------------------------------------------

class ModelCheckpoint(Logger):
    def __init__(
        self, 
        experiment, 
        singleFile=True, 
        bestName=None,
        interval=1000,
        epoch_interval=1
    ):
        self.experiment = experiment
        self.singleFile = singleFile
        self.bestName = bestName
        self.bestValue = None
        self.interval = interval
        self.epoch_interval = epoch_interval
        self.it = 0

    def on_training_end(self):
        self.experiment.save_checkpoint("last.pt")

    def on_epoch_end(self, epoch, stats):        
        isBest = False
        if (self.bestName is None):
            if (self.epoch_interval > 0):
                # Save only at desired epoch intervals
                isBest = (epoch % self.epoch_interval == 0)
            else:
                isBest = True
        else:
            value = stats[self.bestName]
            if (self.bestValue is None):
                isBest = True
                self.bestValue = value
                print("  new best metric: {}: {:.10f}".format(self.bestName, self.bestValue))
            else:
                if (value < self.bestValue):
                    self.bestValue = value
                    print("  new best metric: {}: {:.10f}".format(self.bestName, self.bestValue))
                    isBest = True

        if (isBest):
            if (self.singleFile):
                fn = "checkpoint.pt"
            else:
                fn = f"epoch-{epoch:04d}.pt"

            self.experiment.save_checkpoint(fn)

    def on_iteration(self, epoch, it):
        if (self.interval > 0):
            if (self.it % self.interval == 0):
                fn = f"it-{self.it:07d}.pt"
                self.experiment.save_checkpoint(fn)            
        self.it += 1