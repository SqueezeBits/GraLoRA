# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer

class LastDimWeighting(nn.Module):
    def __init__(self, feature_dim):
        super(LastDimWeighting, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x):
        return x * self.weight

class RasaLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weight
    shared_names = ("rasa_shared_A", "rasa_shared_B")
    adapter_layer_names = ("rasa_A", "rasa_B", "rasa_mixing")
    other_param_names = ("r", "rasa_alpha", "scaling", "rasa_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.effective_r = {}
        self.rasa_k = {}
        self.rasa_alpha = {}
        self.scaling = {}
        self.rasa_dropout = nn.ModuleDict({})

        # Stores a reference to the rasa_A/B ModuleDict.
        # Set to `None` otherwise to avoid computation with random weight
        self.rasa_A = nn.ModuleDict({})
        self.rasa_B = nn.ModuleDict({})
        self.rasa_mixing = nn.ModuleDict({})
        self.rasa_shared_A: Optional[nn.ModuleDict] = None
        self.rasa_shared_B: Optional[nn.ModuleDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        from peft.tuners.vera.buffer_dict import BufferDict

        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names + self.shared_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        module_name,
        rasa_shared_A,
        rasa_shared_B,
        r,
        effective_r,
        rasa_k,
        rasa_alpha,
        rasa_dropout
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.effective_r[adapter_name] = effective_r
        self.rasa_k[adapter_name] = rasa_k
        self.rasa_alpha[adapter_name] = rasa_alpha

        if rasa_dropout > 0.0:
            rasa_dropout_layer = nn.Dropout(p=rasa_dropout)
        else:
            rasa_dropout_layer = nn.Identity()

        self.rasa_dropout.update(nn.ModuleDict({adapter_name: rasa_dropout_layer}))

        # Actual trainable parameters
        self.rasa_A[adapter_name] = nn.Linear(self.in_features, r - rasa_k, bias=False)
        self.rasa_B[adapter_name] = nn.Linear(r - rasa_k, self.out_features, bias=False)
        self.rasa_mixing[adapter_name] = LastDimWeighting(effective_r)
        nn.init.kaiming_uniform_(self.rasa_A[adapter_name].weight, a=math.sqrt(5))
        nn.init.zeros_(self.rasa_B[adapter_name].weight)

        self.rasa_shared_A = rasa_shared_A
        self.rasa_shared_B = rasa_shared_B
        self.module_name = module_name

        # private part
        if r - rasa_k > 0:
            nn.init.constant_(self.rasa_mixing[adapter_name].weight[:r - rasa_k], (0.5 * rasa_alpha) / (r - rasa_k))

        # shared part
        nn.init.constant_(self.rasa_mixing[adapter_name].weight[r - rasa_k:], (0.5 * rasa_alpha) / (effective_r - (r - rasa_k)))

        self.scaling[adapter_name] = 1.0

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

class Linear(nn.Linear, RasaLayer):
    # Rasa implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        module_name,
        rasa_shared_A: nn.ModuleDict,
        rasa_shared_B: nn.ModuleDict,
        r: int = 0,
        effective_r: int = 0,
        rasa_k: int = 0,
        rasa_alpha: int = 1,
        rasa_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        RasaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, module_name, rasa_shared_A, rasa_shared_B, r, effective_r, rasa_k, rasa_alpha, rasa_dropout)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError("Merging is not supported for RasaLayer yet.")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerging is not supported for RasaLayer yet.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("Getting delta weight is not supported for RasaLayer yet.")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.rasa_A.keys():
                    continue
                rasa_A = self.rasa_A[active_adapter]
                rasa_B = self.rasa_B[active_adapter]
                rasa_mixing = self.rasa_mixing[active_adapter]

                if not rasa_A.weight.requires_grad:
                    # there might be a more elegant way to do this
                    self.rasa_shared_A[active_adapter][self.module_name] = self.rasa_shared_A[active_adapter][self.module_name].to(rasa_A.weight.device)
                    self.rasa_shared_B[active_adapter][self.module_name] = self.rasa_shared_B[active_adapter][self.module_name].to(rasa_B.weight.device)

                rasa_shared_A = self.rasa_shared_A[active_adapter][self.module_name]
                rasa_shared_B = self.rasa_shared_B[active_adapter][self.module_name]

                dropout = self.rasa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = x.to(rasa_A.weight.dtype)
                result += F.linear(
                    rasa_mixing(
                        F.linear(
                            dropout(x),
                            torch.cat([rasa_A.weight, rasa_shared_A.weight], dim=0),
                        )
                    ),
                    torch.cat([rasa_B.weight, rasa_shared_B.weight], dim=1)
                ) * scaling

            result = result.to(torch_result_dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rasa." + rep
