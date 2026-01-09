import jittor
import jittor.nn as nn
from typing import Dict, Optional, Tuple, Union


def build_norm_layer(cfg, num_features, postfix=''):
    assert isinstance(cfg, dict) and 'type' in cfg
    norm_type = cfg['type']
    requires_grad = cfg.get('requires_grad', True)
    eps = cfg.get('eps', 1e-5)
    momentum = cfg.get('momentum', 0.1)

    if norm_type == 'BN':
        layer = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
    elif norm_type == 'SyncBN':
        layer = nn.SyncBatchNorm(num_features, eps=eps, momentum=momentum)
    elif norm_type == 'GN':
        num_groups = cfg.get('num_groups', 32)
        layer = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, eps=eps)
    elif norm_type == 'LN':
        layer = nn.LayerNorm(num_features, eps=eps)
    else:
        raise ValueError(f'Unrecognized norm type {norm_type}')

    for param in layer.parameters():
        param.requires_grad = requires_grad

    name = 'norm' + (str(postfix) if postfix else '')
    return name, layer


def build_activation_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    assert isinstance(cfg, dict) and 'type' in cfg, \
        "The cfg must be a dict and contain the key 'type'."

    act_type = cfg['type']
    cfg_ = cfg.copy()
    cfg_.pop('type')  # 剩下的作为参数传入激活函数构造器

    # 激活层映射字典
    act_layers = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'ELU': nn.ELU,
        'Softmax': nn.Softmax,
        'Softplus': nn.Softplus,
        'Identity': nn.Identity
    }

    if act_type not in act_layers:
        raise KeyError(f"Unsupported activation type: {act_type}")
    layer_cls = act_layers[act_type]
    return layer_cls()


def build_conv_layer(cfg, *args, **kwargs):
    """构建卷积层（支持多类型）"""
    if cfg is None:
        conv_type = 'Conv2d'
        layer_args = {}
    else:
        assert isinstance(cfg, dict) and 'type' in cfg, \
            "cfg must be a dict containing the key 'type'"
        conv_type = cfg['type']
        layer_args = cfg.copy()
        layer_args.pop('type')

    # 支持的卷积类型
    conv_layers = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'ConvTranspose3d': nn.ConvTranspose3d,
    }

    if conv_type not in conv_layers:
        raise KeyError(f"Unsupported conv type: {conv_type}")

    conv_cls = conv_layers[conv_type]
    return conv_cls(*args, **layer_args, **kwargs)


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 order: tuple = ('conv', 'norm', 'act')
                 ):
        super().__init__()
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.order = order
        conv_cfg = None
        conv_padding = 0
        dilation = 1
        groups = 1
        bias = not self.with_norm

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)  # type: ignore
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None  # type: ignore

        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.activate = build_activation_layer(act_cfg_)

        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            jittor.init.kaiming_normal_(self.conv.weight, a=a, nonlinearity=nonlinearity)

    def execute(self,
                x: jittor.Var,
                activate: bool = True,
                norm: bool = True) -> jittor.Var:
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)

        return x