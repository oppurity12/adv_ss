import torch
import torch.nn as nn

from .tasnet import ConvTasNet
from torchmetrics import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SignalNoiseRatio


class Attack(object):
  def __init__(self, name, model):
    self.name = name
    self.model = model
    self.device = next(model.parameters()).device

  def forward(self, *input):
          r"""
          It defines the computation performed at every call.
          Should be overridden by all subclasses.
          """
          raise NotImplementedError


class PGD(Attack):
  def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
    
  def forward(self, audio, target_A='drums', target_B='vocals', loss_f='l1'):
    SOURCES = {"drums": 0, "bass": 1, "other":2, "vocals":3}

    criterion = nn.L1Loss() if loss_f == 'l1' else nn.MSELoss()

    assert isinstance(self.model, ConvTasNet)

    self.model.eval

    target_idx_1 = SOURCES[target_A]
    target_idx_2 = SOURCES[target_B]

    self.model.attack = True

    est_source, est_mask = self.model(audio)

    est_mask = est_mask[target_idx_2]

    adv_audio = audio.clone().detach()

    for _ in range(self.steps):
      adv_audio.requires_grad = True
      adv_est_source, adv_est_mask = self.model(audio)
      adv_est_mask = adv_est_mask[target_idx_1]

      cost = -criterion(adv_est_mask, est_mask)
      grad = torch.autograd.grad(cost, adv_images,
                                      retain_graph=False, create_graph=False)[0]

      adv_audio = adv_audio.detach() + self.alpha*grad.sign()
      delta = torch.clamp(adv_audio - audio, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(audio + delta, min=0, max=1).detach()
    
    return adv_est_source
