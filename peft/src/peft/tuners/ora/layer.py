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

import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class OraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("ora_A", "ora_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.ora_dropout = nn.ModuleDict({})

        # Stores a reference to the ora_A/B ModuleDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.ora_A: Optional[nn.ModuleDict] = None
        self.ora_B: Optional[nn.ModuleDict] = None

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

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        layer_id,
        module_name,
        ora_indices_A,
        ora_indices_B,
        ora_A: nn.ModuleDict,
        ora_B: nn.ModuleDict,
        r,
        ora_dropout,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if ora_dropout > 0.0:
            ora_dropout_layer = nn.Dropout(p=ora_dropout)
        else:
            ora_dropout_layer = nn.Identity()

        self.ora_dropout.update(nn.ModuleDict({adapter_name: ora_dropout_layer}))
        self.layer_id = layer_id
        self.module_name = module_name
        self.ora_indices_A = ora_indices_A
        self.ora_indices_B = ora_indices_B
        self.ora_A = ora_A
        self.ora_B = ora_B

        if adapter_name not in ora_A:
            raise NotImplementedError("Adapter name not found in the provided ora_A")

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

class Linear(nn.Linear, OraLayer):
    # Ora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        layer_id,
        module_name,
        ora_indices_A,
        ora_indices_B,
        ora_A: nn.ModuleDict,
        ora_B: nn.ModuleDict,
        adapter_name: str,
        r: int = 0,
        ora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        OraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, layer_id, module_name, ora_indices_A, ora_indices_B, ora_A, ora_B, r, ora_dropout)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError("Merging is not supported for OraLayer yet.")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerging is not supported for OraLayer yet.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("Getting delta weight is not supported for OraLayer yet.")

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
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ora_A.keys():
                    continue

                ora_A = self.ora_A[active_adapter][self.module_name]
                ora_B = self.ora_B[active_adapter][self.module_name]
                ora_indices_A = self.ora_indices_A[active_adapter][self.layer_id][self.module_name]
                ora_indices_B = self.ora_indices_B[active_adapter][self.layer_id][self.module_name]

                dropout = self.ora_dropout[active_adapter]
                x = x.to(ora_A.weight.dtype)
                result = result + F.linear(F.linear(dropout(x), ora_A.weight[:, ora_indices_A]), ora_B.weight[ora_indices_B, :])

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ora." + rep
