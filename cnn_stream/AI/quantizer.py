"""
Author: Foivos Tsimpourlas
"""
import torch
import copy

from typing import Union

"""
TASK 2 module.
"""

def halve(m: Union[torch.nn.Module, torch.Tensor]) -> Union[torch.nn.Module, torch.Tensor]:
  """
  Return fp16 version of either model or tensor.
  """
  return m.half()

def quantize_dynamic(m: torch.nn.Module) -> torch.nn.Module:
  """
  Dynamic Quantization mode.
  """
  if isinstance(m, torch.Tensor):
    return m
  return torch.quantization.quantize_dynamic(
    model = m, dtype = torch.qint8, inplace = False
  )

def quantize_fx(m: torch.nn.Module) -> torch.nn.Module:
  """
  FX Quantization mode.
  """
  if isinstance(m, torch.Tensor):
    return m
  from torch.quantization import quantize_fx
  qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
  model_prepared = quantize_fx.prepare_fx(m, qconfig_dict)
  return quantize_fx.convert_fx(model_prepared)
