"""
Implementations of several autoregressive models described in 

https://arxiv.org/pdf/1601.06759.pdf
"""

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn import Conv2d
from typing import Sequence
from dataclasses import dataclass
from typing import Literal
import einops


class LSTMCell(nn.Module):
    """Implements a very simple LSTM cell with input-to-state and state-to-state components"""

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.W_is = nn.Linear(n_features, 4 * n_features)
        self.W_ss = nn.Linear(n_features, 4 * n_features)

    def forward(
        self,
        X,
        h=None,
        c=None,
    ):
        if h is None:
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)
        if c is None:
            c = torch.zeros(X.shape[0], self.n_features, device=X.device)

        fiog = self.W_is(X) + self.W_ss(h)
        f, i, o, g = torch.split(fiog, self.n_features, dim=-1)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

    def __call__(self, X, c=None, h=None):
        """describes the way the module is called"""
        return super([X], dict(c=c, h=h))


class MaskedConvolution2D(nn.Module):
    """
    Implements masked convolution as in the PixelRNN paper.

    Typical convolutions allow communications with all features of neighboring pixels,
    but by masking we can restrict communications to only neighbors within a desired context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        mask_type="B",
    ):
        super().__init__()
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if not isinstance(kernel_size, Sequence):
            kernel_size = [kernel_size]
        for d in kernel_size:
            if d % 2 == 0:
                raise ValueError(
                    "This layer only works if all kernel sizes are odd numbers."
                )

        mask = torch.ones_like(self.conv2d.weight)
        C_out, C_in, H, W = mask.shape
        # mask all the context to the left and below, including the middle
        center_y = H // 2
        center_x = W // 2
        mask[:, :, center_y + 1 :, :] = 0
        mask[:, :, :, center_x + 1 :] = 0

        if mask_type == "A":
            if C_in // 3:
                C_in_color = C_in // 3
                C_out_color = C_out // 3

                # channels: RGB

                # first mask all between pixel connections
                mask[:, :, center_y, center_x] = 0

                # G can look at R
                mask[C_out_color : C_out_color * 2, :C_in_color, center_y, center_x] = 1

                # B can look at R and G
                mask[
                    C_out_color * 2 : C_out_color * 3,  # B output channel
                    : C_in_color * 2,  # R and G input channel
                    center_y,
                    center_x,
                ] = 1

            if C_in == 1:
                # Pixel can't look at itself
                mask[:, :, center_y, center_x] = 0

        else: 
            if C_in // 3:
                C_in_color = C_in // 3
                C_out_color = C_out // 3

                # channels: RGB

                # first mask all between pixel connections
                mask[:, :, center_y, center_x] = 0

                # R can look at R 
                mask[:C_out_color, :C_in_color, center_y, center_x] = 1 

                # G can look at R and G
                mask[C_out_color : C_out_color * 2, :C_in_color*2, center_y, center_x] = 1

                # B can look at everything
                mask[
                    C_out_color * 2 : C_out_color * 3,  
                    :,  
                    center_y,
                    center_x,
                ] = 1

        self.register_buffer("mask", mask)

    def forward(self, X):
        self.conv2d.weight.data = self.conv2d.weight.data * self.mask
        return self.conv2d(X)


class PixelRNNLayer(nn.Module):
    """
    Input: 1, c, h, w
    Output: 1, c, h, w
    """

    def __init__(self, in_features, out_features, row_kernel_size=3, rgb_input=True):
        super().__init__()
        self.input_to_state = MaskedConvolution2D(
            in_features,
            4 * out_features,
            (1, row_kernel_size),
            stride=1,
            padding=(0, 1),
            mask_type='B'
        )
        self.state_to_state = torch.nn.Conv1d(
            in_channels=out_features,
            out_channels=4 * out_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.rgb_input = rgb_input

    def forward(self, X, h=None, c=None):
        
        B, C, H, W = X.shape
        input_to_state_whole_image = self.input_to_state(X)  # B, C, H, W

        row_features = []
        h = torch.zeros(
            B, self.out_features, W, device=X.device
        )  # initially state is zero
        c = torch.zeros(B, self.out_features, W, device=X.device)

        for i in range(H):
            state_to_state = self.state_to_state(h)

            input_to_state = input_to_state_whole_image[:, :, i, :]
            gates = state_to_state + input_to_state

            if self.rgb_input: 
                r, g, b = torch.split(gates, gates.shape[1]//3, 1)
                f_r, i_r, o_r, g_r = torch.split(r, r.shape[1]//4, 1)
                f_g, i_g, o_g, g_g = torch.split(g, g.shape[1]//4, 1)
                f_b, i_b, o_b, g_b = torch.split(b, b.shape[1]//4, 1)

                f = torch.cat(
                    [f_r, f_g, f_b], dim=1
                )
                i = torch.cat(
                    [i_r, i_g, i_b], dim=1
                )
                o = torch.cat(
                    [o_r, o_g, o_b], dim=1
                )
                g = torch.cat(
                    [g_r, g_g, g_b], dim=1
                )
            else: 
                f, i, o, g = torch.split(gates, self.out_features, dim=1)

            f = f.sigmoid()
            i = i.sigmoid()
            o = o.sigmoid()
            g = g.tanh()

            c = f * c + i * g
            h = o * c.tanh()

            row_features.append(h)

        return torch.stack(row_features, -2)


class PixelRNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=256 * 3,
        hidden_dim=3 * 32,
        n_layers=4,
        use_residual=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.out_channels = out_channels

        self.input_conv = MaskedConvolution2D(
            in_channels=in_channels,
            out_channels=self.hidden_dim,
            kernel_size=(7, 7),
            padding=3,
            mask_type="A",
        )

        self.rnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.rnn_layers.append(
                PixelRNNLayer(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    row_kernel_size=3,
                    rgb_input=in_channels==3,
                )
            )

        self.output_conv = MaskedConvolution2D(
            in_channels=self.hidden_dim,
            out_channels=self.out_channels,
            kernel_size=(1, 1),
        )

    def forward(self, X):
        X = self.input_conv(X).relu()

        for i, layer in enumerate(self.rnn_layers):
            identity = X
            X = layer(X)
            if self.use_residual:
                X = X + identity

        X = self.output_conv(X)

        return X


class ConditionalPixelRNN(PixelRNN):
    ...


if __name__ == "__main__":
    ...
    # model = PixelRNN(in_channels=1, hidden_dim=32, n_layers=1)
    # input_data = torch.randn(1, 1, 28, 28)
    # input_data.requires_grad = True
# 
    conv = MaskedConvolution2D(
        in_channels=3, out_channels=6, kernel_size=(3, 3), mask_type="A"
    )
    breakpoint()