import torch
import torch.nn as nn
import lib.layers as layers
from lib.layers.odefunc import ODEnet
from lib.layers.squeeze import squeeze, unsqueeze
import torchvision.transforms as tforms
import numpy as np


class ODENVP(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.
    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
            self,
            input_size,
            n_scale=float('inf'),
            n_blocks=2,
            strides=None,
            intermediate_dims=(32,),
            nonlinearity="softplus",
            layer_type="concat",
            squash_input=True,
            squeeze_first=False,
            zero_last=True,
            div_samples=1,
            alpha=0.05,
            cnf_kwargs=None, reduce_dim_first=False
    ):
        super(ODENVP, self).__init__()
        if squeeze_first:
            bsz, c, w, h = input_size
            c, w, h = c * 4, w // 2, h // 2
            input_size = bsz, c, w, h
        if reduce_dim_first:
            bsz, c, w, h = input_size
            _, w, h = c * 4, w // 2, h // 2
            reduced_input = bsz, c, w, h
            self.n_scale = min(n_scale, self._calc_n_scale(reduced_input))
        else:
            self.n_scale = min(n_scale, self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.layer_type = layer_type
        self.zero_last = zero_last
        self.div_samples = div_samples
        self.nonlinearity = nonlinearity
        self.strides = strides
        self.squash_input = squash_input
        self.alpha = alpha
        self.squeeze_first = squeeze_first
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}
        self.transforms_2 = tforms.Compose([tforms.Resize(32)])
        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net_complete(input_size)
        # self._load_weights()
        self._load_complete_state_dict()
        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

    def _load_complete_state_dict(self):
        state_dict = torch.load(
            "/HPS/CNF/work/ffjord-rnode/experiments/celebahq/example/intermediate.pth")["state_dict"]
        skip_index = len("transforms.")
        loaded_state_dict = {}
        for key in state_dict.keys():
            number = str(int(key[skip_index]))
            loaded_state_dict[number + key[skip_index + 1:]] = state_dict[key]
        self.transforms.load_state_dict(loaded_state_dict)

    def _load_weights(self):
        state_dict = torch.load(
            "/HPS/CNF/work/ffjord-rnode/experiments/celebahq/example/32_simple_experiment/best.pth")["state_dict"]

        skip_index = len("transforms.")
        loaded_state_dict = {}
        for key in state_dict.keys():
            number = str(int(key[skip_index]) + 1)
            loaded_state_dict[number + key[skip_index + 1:]] = state_dict[key]
        final_state_dict = {}
        for key in self.transforms.state_dict().keys():
            if key[0] == "0":
                final_state_dict[key] = self.transforms.state_dict()[key]
            else:
                try:
                    final_state_dict[key] = loaded_state_dict[key]
                except:
                    continue

        for key in self.transforms.state_dict().keys():
            if key not in final_state_dict.keys():
                final_state_dict[key] = self.transforms.state_dict()[key]
        self.transforms.load_state_dict(final_state_dict)

    def _build_net_complete(self, input_size):
        bsz, c, h, w = input_size
        transforms = []
        transforms.append(
            StackedCNFLayers(
                initial_size=(c, h, w),
                div_samples=self.div_samples,
                zero_last=self.zero_last,
                layer_type=self.layer_type,
                strides=self.strides,
                idims=self.intermediate_dims,
                squeeze=(0 < self.n_scale - 1),  # don't squeeze last layer
                init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform())
                if self.squash_input else None,
                n_blocks=self.n_blocks,
                cnf_kwargs=self.cnf_kwargs,
                nonlinearity=self.nonlinearity,
            )
        )
        c, h, w = c, h // 2, w // 2
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    div_samples=self.div_samples,
                    zero_last=self.zero_last,
                    layer_type=self.layer_type,
                    strides=self.strides,
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    div_samples=self.div_samples,
                    zero_last=self.zero_last,
                    layer_type=self.layer_type,
                    strides=self.strides,
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform())
                    if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, sharing_factor, logpx=None, reg_states=tuple(), reverse=False):
        if reverse:
            out = self._generate(x, logpx, reg_states)
            if self.squeeze_first:
                x = unsqueeze(out[0])
            else:
                x = out[0]
            return x, out[1], out[2]
        else:
            if self.squeeze_first:
                x = squeeze(x)
            return self._logdensity(x, sharing_factor, logpx, reg_states)

    def _logdensity(self, x, sharing_factor, logpx=None, reg_states=tuple()):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        image_2 = self.transforms_2(x)
        x, _logpx, reg_states = self.transforms[0].forward(x, _logpx, reg_states)
        d = x.size(1) // 2
        x, factor_out = x[:, :d], x[:, d:]
        out = [factor_out]
        d = x.size(1) // 2
        x1, x2 = x[:, :d], x[:, d:]
        x1 = x1 * sharing_factor + image_2 * (1 - sharing_factor)
        x2 = x2 * sharing_factor + image_2 * (1 - sharing_factor)
        _logpx1 = _logpx
        _logpx2 = _logpx
        reg_states1 = reg_states
        reg_states2 = reg_states
        for idx in range(1, len(self.transforms)):

            x1, _logpx1, reg_states1 = self.transforms[idx].forward(x1, _logpx1, reg_states1)
            x2, _logpx2, reg_states2 = self.transforms[idx].forward(x2, _logpx2, reg_states2)
            if idx < len(self.transforms) - 1:
                d = x1.size(1) // 2
                x1, factor_out1 = x1[:, :d], x1[:, d:]
                d = x2.size(1) // 2
                x2, factor_out2 = x2[:, :d], x2[:, d:]
            else:
                # last layer, no factor out
                factor_out1 = x1
                factor_out2 = x2

            out.append(torch.cat((factor_out1, factor_out2), 1))
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        if len(reg_states1) == 0 or len(reg_states2) == 0:
            return out, (_logpx1 + _logpx2) / 2.0, ()
        return out, (_logpx1 + _logpx2) / 2.0, ((reg_states1[0] + reg_states2[0]) / 2.0,
                                                (reg_states1[1] + reg_states2[1]) / 2.0)

    def _generate(self, z, logpz=None, reg_states=tuple()):
        z = z.view(z.shape[0], -1)
        zs = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims)
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz, _ = self.transforms[-1](zs[-1], _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz, reg_states = self.transforms[idx](z_prev, _logpz, reg_states, reverse=True)
        return z_prev, _logpz, reg_states


class StackedCNFLayers(layers.SequentialFlow):
    def __init__(
            self,
            initial_size,
            idims=(32,),
            nonlinearity="softplus",
            layer_type="concat",
            div_samples=1,
            squeeze=True,
            init_layer=None,
            n_blocks=1,
            zero_last=True,
            strides=None,
            cnf_kwargs={},
    ):
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = ODEnet(idims, size, strides, True, layer_type=layer_type, nonlinearity=nonlinearity,
                         zero_last_weight=zero_last)
            f = layers.ODEfunc(net, div_samples=div_samples)
            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]
            post = [layers.CNF(_make_odefunc(after_squeeze_size), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [layers.SqueezeLayer(2)] + post
        else:
            chain += [layers.CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]

        super(StackedCNFLayers, self).__init__(chain)
