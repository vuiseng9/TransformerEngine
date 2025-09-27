from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import NVFP4BlockScaling, Recipe
from ..constants import NVFP4_BLOCK_SCALING_SIZE
from ..utils import devices_match, round_up_to_nearest_multiple
from .quantized_tensor import QuantizedTensor, Quantizer
from .mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

aten = torch.ops.aten

class NVFP4Quantizer(MXFP8Quantizer):
    # no override on 
    # calibrate
    def __init__(
        self,
        fp8_dtype: TE_DType = TE_DType.kFloat4E2M1,  # default to NVFP4
        *,
        rowwise: bool = True,
        columnwise: bool = True,
        mxfp8_bw_quantize: bool = False,
    ) -> None:
        super().__init__(fp8_dtype, rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.mxfp8_bw_quantize = mxfp8_bw_quantize
        if mxfp8_bw_quantize:
            self.mxfp8_quantizer = MXFP8Quantizer(fp8_dtype=TE_DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)

    def quantize(self, tensor, *, out = None, dtype = None):
        nvfp4_quantized = super().quantize(tensor, out=out, dtype=dtype)
        if self.mxfp8_bw_quantize:
            # Use MXFP8 quantizer for backward pass quantization
            nvfp4_quantized.bw_quantizer = self.mxfp8_quantizer
            nvfp4_quantized.bw_tensor    = self.mxfp8_quantizer(tensor)
        return nvfp4_quantized

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, NVFP4Tensor), f"Cannot store quantized NVFP4 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        # Update FP8 dtype
        dst._fp8_dtype = self.dtype

        return dst

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> NVFP4Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert (
            shape[-1] % NVFP4_BLOCK_SCALING_SIZE == 0
            and math.prod(shape[:-1]) % NVFP4_BLOCK_SCALING_SIZE == 0
        ), (
            f"Incorrect shape {shape} for NVFP4. Tensor dims must be divisible by"
            f" {NVFP4_BLOCK_SCALING_SIZE}"
        )

        # Allocate FP8 data TODO(VS), do we pack fp4
        data = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_inv = torch.zeros(
            round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
            round_up_to_nearest_multiple(shape[-1] // NVFP4_BLOCK_SCALING_SIZE, 4),
            dtype=torch.uint8,
            device=device,
        )

        # Allocate FP8 data transpose if needed TODO(VS), do we pack fp4
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty_like(data)
            columnwise_scale_inv = torch.zeros(
                round_up_to_nearest_multiple(math.prod(shape[:-1]) // NVFP4_BLOCK_SCALING_SIZE, 4),
                round_up_to_nearest_multiple(shape[-1], 128),
                dtype=torch.uint8,
                device=device,
            )

        # Construct NVFP4 tensor
        return NVFP4Tensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return NVFP4BlockScaling

class NVFP4Tensor(MXFP8Tensor):
    # no override on 
    # quantize_
    # dequantize (low priority)
    # clone
    # view
    # reshape
    # contiguous

    def __repr__(self, *, tensor_contents=None):
        return f"NVFP4Tensor(fp4_dtype={self._fp8_dtype}, data={self.dequantize(dtype=self.dtype)})"

    def _get_quantizer(self) -> Quantizer:
        """Get builder for quantized tensor
        Quantizer can be used for in-place operations.
        """
        if self._quantizer is not None:
            return self._quantizer
        return NVFP4Quantizer(
            fp8_dtype=self._fp8_dtype,
        )

    def detach(self) -> NVFP4Tensor:
        # pylint: disable=missing-function-docstring
        # TODO(ksivamani): Fix the detach bug
        return NVFP4Tensor.make_like(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            return NVFP4Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                quantizer=tensor._quantizer,
                requires_grad=False,
                fp8_dtype=tensor._fp8_dtype,
            )

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
        shape: torch.shape,
    ) -> NVFP4Tensor:
        """Build NVFP4Tensor, for use in __reduce__
        __reduce_ex__ assumes object constructor has positional
        arguments.
        """
        return NVFP4Tensor(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            shape=shape,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            NVFP4Tensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self.shape,
            ),
        )

    def _get_data(self) -> NVFP4Tensor:
        """Get tensor data property"""
        return super()._get_data()

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property
        Just takes FP8 data if setting from a MXFP8Tensor. Otherwise
        casts to FP8.
        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is MXFP8Tensor
        if isinstance(tensor, NVFP4Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    NVFP4Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(NVFP4Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._fp8_dtype = tensor._fp8_dtype
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting MXFP8Tensor.data
    data = property(_get_data, _set_data)