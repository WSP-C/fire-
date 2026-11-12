
import torch
import torch.nn as nn

from .block import C2f


class EMA(nn.Module):
    """Efficient multi-scale attention with an identity-initialized residual gate."""

    def __init__(self, channels, factor=8):
        super().__init__()

        if channels < 1 or factor < 1:
            raise ValueError(f"channels and factor must be positive, got channels={channels}, factor={factor}")

        # Select the largest requested group count that divides the channels.
        groups = min(channels, factor)

        while channels % groups != 0:
            groups -= 1

        self.groups = groups

        group_channels = channels // groups

        self.softmax = nn.Softmax(dim=-1)

        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(
            group_channels,
            group_channels
        )

        self.conv1x1 = nn.Conv2d(
            group_channels,
            group_channels,
            kernel_size=1
        )

        self.conv3x3 = nn.Conv2d(
            group_channels,
            group_channels,
            kernel_size=3,
            padding=1
        )

        # Start as an exact identity mapping so pretrained features are not
        # attenuated before the attention branch has learned useful weights.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        b, c, h, w = x.size()

        group_x = x.reshape(
            b * self.groups,
            c // self.groups,
            h,
            w
        )

        # Pool along height and width independently.
        x_h = self.pool_h(group_x)

        x_w = self.pool_w(group_x).permute(
            0, 1, 3, 2
        )

        # Fuse directional context.
        hw = self.conv1x1(
            torch.cat([x_h, x_w], dim=2)
        )

        x_h, x_w = torch.split(
            hw,
            [h, w],
            dim=2
        )

        # Direction-aware branch.
        x1 = self.gn(
            group_x
            * x_h.sigmoid()
            * x_w.permute(0, 1, 3, 2).sigmoid()
        )

        # Local-context branch.
        x2 = self.conv3x3(group_x)

        # Cross-spatial interaction.
        x11 = self.softmax(
            self.agp(x1)
            .reshape(b * self.groups, -1, 1)
            .permute(0, 2, 1)
        )

        x12 = x2.reshape(
            b * self.groups,
            c // self.groups,
            h * w
        )

        x21 = self.softmax(
            self.agp(x2)
            .reshape(b * self.groups, -1, 1)
            .permute(0, 2, 1)
        )

        x22 = x1.reshape(
            b * self.groups,
            c // self.groups,
            h * w
        )

        weights = (
            torch.matmul(x11, x12)
            + torch.matmul(x21, x22)
        )

        weights = weights.reshape(
            b * self.groups,
            1,
            h,
            w
        )

        attention = weights.sigmoid()
        out = group_x + self.gamma * group_x * attention
        return out.reshape(b, c, h, w)


class C2fEMA(C2f):
    """C2f followed by residual EMA without changing YOLO layer indices."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, factor=8):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.ema = EMA(c2, factor)

    def forward(self, x):
        return self.ema(super().forward(x))

    def forward_split(self, x):
        return self.ema(super().forward_split(x))


class EMACentered(EMA):
    """EMA with centered attention and an identity-initialized per-channel residual gate."""

    def __init__(self, channels, factor=8):
        super().__init__(channels, factor)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat((x_h, x_w), dim=2))
        x_h, x_w = torch.split(hw, (h, w), dim=2)

        directional = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        local = self.conv3x3(group_x)
        directional_query = self.softmax(self.agp(directional).flatten(2).transpose(1, 2))
        local_query = self.softmax(self.agp(local).flatten(2).transpose(1, 2))
        directional_values = local.flatten(2)
        local_values = directional.flatten(2)
        attention = torch.matmul(directional_query, directional_values) + torch.matmul(local_query, local_values)
        attention = attention.reshape(b * self.groups, 1, h, w).sigmoid()

        # Centering around zero avoids the amplification-only bias of sigmoid gating.
        delta = (group_x * (2.0 * attention - 1.0)).reshape(b, c, h, w)
        return x + self.gamma * delta


class C2fEMACentered(C2f):
    """C2f followed by centered residual EMA for context-rich feature levels."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, factor=8):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.ema = EMACentered(c2, factor)

    def forward(self, x):
        return self.ema(super().forward(x))

    def forward_split(self, x):
        return self.ema(super().forward_split(x))
