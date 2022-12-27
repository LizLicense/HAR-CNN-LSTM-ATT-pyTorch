import torch
import torch.nn.functional as F

import os
import collections
import pandas as pd
import numpy as np




def s_update(self, samples):
    # ====== Data =====================
    data = samples['sample_ori'].float()
    labels = samples['class_labels'].long()

    # ====== Source =====================
    self.optimizer.zero_grad()

    # Src original features
    features = self.feature_extractor(data)
    features = self.temporal_encoder(features)
    logits = self.classifier(features)

    # Cross-Entropy loss
    x_ent_loss = self.cross_entropy(logits, labels)

    x_ent_loss.backward()
    self.optimizer.step()

    return {'Total_loss': x_ent_loss.item()}, \
            [self.feature_extractor, self.temporal_encoder, self.classifier]

