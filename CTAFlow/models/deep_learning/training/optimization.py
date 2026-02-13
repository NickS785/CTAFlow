import numpy as np
import torch
import pandas as pd


class AccuracyWeightedPnL:

    def __init__(self, max_class_wt, accuracy_wt=0.8) -> None:
        if max_class_wt <= 1.0:
            max_class_wt *= 100
        self.max_class_wt = max_class_wt
        self.accuracy_wt = accuracy_wt
        self.loss_wt = 1.0 - accuracy_wt

    def score(self, pnl, loss, accuracy):

        if accuracy <= 1.0:
            accuracy *= 100
        if accuracy - self.max_class_wt >= 2.0:
            pnl_multi = (accuracy - self.max_class_wt) * self.accuracy_wt
        else:
            pnl_multi = (accuracy / self.max_class_wt) * self.accuracy_wt

        score = (pnl * pnl_multi) / (loss * self.loss_wt)

        return score
