import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
import torch.utils.model_zoo as model_zoo

import warnings

from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import  Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _make_divisible, _ovewrite_named_param, handle_legacy_interface





BN_MOMENTUM = 0.1


__all__ = ["MobileNetV2", "MobileNet_V2_Weights", "mobilenet_v2"]

model_urls = {
    'MobileNetV2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# necessary for backwards compatibility
class _DeprecatedConvBNAct(Conv2dNormActivation):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The ConvBNReLU/ConvBNActivation classes are deprecated since 0.12 and will be removed in 0.14. "
            "Use torchvision.ops.misc.Conv2dNormActivation instead.",
            FutureWarning,
        )
        if kwargs.get("norm_layer", None) is None:
            kwargs["norm_layer"] = nn.BatchNorm2d
        if kwargs.get("activation_layer", None) is None:
            kwargs["activation_layer"] = nn.ReLU6
        super().__init__(*args, **kwargs)


ConvBNReLU = _DeprecatedConvBNAct
ConvBNActivation = _DeprecatedConvBNAct


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 insted of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        heads,
        head_conv,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        
        self.inplanes = 320
        self.heads = heads
        self.deconv_with_bias = False
        
        
        super().__init__()
        _log_api_usage_once(self)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 320
        # last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, norm_layer=norm_layer, activation_layer=nn.ReLU6)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            Conv2dNormActivation(
                input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6
            )
        )
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(self.last_channel, num_classes),
        # )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [40, 40, 40],
            [4, 4, 4],
        )
        # self.final_layer = []

        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(40, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=40,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)  
          
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding         
                
    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)     

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]
        
        # Cannot use "squeeze" as batch-size can be 1
        # x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    
    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    # print('=> init {}.weight as 1'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


_COMMON_META = {
    "num_params": 3504872,
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class MobileNet_V2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", MobileNet_V2_Weights.IMAGENET1K_V1))
def mobilenet_v2(
    *, heads, head_conv, weights: Optional[MobileNet_V2_Weights] = None, progress: bool = True, **kwargs: Any
) -> MobileNetV2:

    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV2(heads, head_conv, width_mult = 1.0, **kwargs)
    model.init_weights(pretrained = True)

    # if weights is not None:
    #     model.load_state_dict(weights.get_state_dict(progress=progress))

    return model





def get_pose_net(num_layers,heads,head_conv):


    model = mobilenet_v2( heads = heads , head_conv=head_conv)
  
  
    # url = model_urls['MobileNetV2']
    # pretrained_state_dict_all = model_zoo.load_url(url)
    # # print(pretrained_state_dict_all.keys())
    # pretrained_state_dict = {k:v for k,v in pretrained_state_dict_all.items() if model.state_dict()[k].numel() == v.numel() }
    # # print(pretrained_state_dict.keys())
    # print('=> loading pretrained model {}'.format(url))
    # model.load_state_dict(pretrained_state_dict, strict=False)


    url = model_urls['MobileNetV2']
    pretrained_state_dict_all = model_zoo.load_url(url)
    # print(model.state_dict().keys())
    # print("------------------------------------")
    # print(pretrained_state_dict_all.keys())

    
    # Skip pretrained weights that are not suitable for the defined model.
    pretrained_state_dict = {k: pretrained_state_dict_all[k] for k in model.state_dict().keys() 
                            if  k in pretrained_state_dict_all and 
                            pretrained_state_dict_all[k].size() == model.state_dict()[k].size()}
    # print(pretrained_state_dict.keys())
    # print(len(pretrained_state_dict.keys()))
    # print('=> loading pretrained model {}'.format(url))
    model.load_state_dict(pretrained_state_dict, strict=False)
    # print(model.state_dict().keys())
    # print(len(model.state_dict().keys()))
    
    # print(model.state_dict().values())

    return model

