# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch.nn as nn

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from models.resnet import ResNetBase


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3):
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        # define self-sup heads
        self.proj_dim = self.PLANES[7] * self.BLOCK.expansion

        # projector head
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(self.proj_dim))

        # predictor head
        self.predictor = nn.Sequential(
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=D))

    def _forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_bottle = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out_bottle)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        return out, out_bottle

    def _forward_heads(self, x):
        out_seg = self.final(x)
        out_en = self.encoder(x)
        out_pred = self.predictor(out_en)
        return out_seg, out_en, out_pred

    def forward(self, x, is_train=True):
        if is_train:
            x0, x1 = x

            # future: t0 -> t1
            out_backbone0, out_bottle0 = self._forward(x0)

            # past: t1 -> t0
            out_backbone1, out_bottle1 = self._forward(x1)

            # future pred
            out_seg0, out_en0, out_pred0 = self._forward_heads(out_backbone0)
            # past pred
            out_seg1, out_en1, out_pred1 = self._forward_heads(out_backbone1)

            return out_seg0.F, out_en0.F, out_pred0.F, out_backbone0.F, out_bottle0, \
                   out_seg1.F, out_en1.F, out_pred1.F, out_backbone1.F, out_bottle1
        else:
            # forward in backbone
            out_backbone, out_bottle = self._forward(x)

            # forward in final
            out_seg = self.final(out_backbone)

            return out_seg.F, out_backbone.F, out_bottle


class MinkUNet18_SSL(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet18_HEADS(nn.Module):

    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

        # define self-sup heads
        self.proj_dim = self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion

        self.relu = ME.MinkowskiReLU(inplace=True)

        # projector head
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(
                self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim))

        # self.encoder = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
        #         self.proj_dim,
        #         kernel_size=1,
        #         bias=True,
        #         dimension=self.seg_model.D),
        #     ME.MinkowskiBatchNorm(self.proj_dim),
        #     self.relu,
        #     ME.MinkowskiConvolution(
        #         self.proj_dim,
        #         self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
        #         kernel_size=1,
        #         bias=True,
        #         dimension=self.seg_model.D))

        # predictor head
        self.predictor = nn.Sequential(
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D))

    def _forward_heads(self, x):
        out_seg = self.seg_model.final(x)
        out_en = self.encoder(x)
        out_pred = self.predictor(out_en)
        return out_seg, out_en, out_pred

    def forward(self, x, is_train=True):
        if is_train:
            x0, x1 = x

            # future: t0 -> t1
            out_backbone0, out_bottle0 = self.seg_model(x0, is_seg=False)

            # past: t1 -> t0
            out_backbone1, out_bottle1 = self.seg_model(x1, is_seg=False)

            # future pred
            out_seg0, out_en0, out_pred0 = self._forward_heads(out_backbone0)
            # past pred
            out_seg1, out_en1, out_pred1 = self._forward_heads(out_backbone1)

            return out_seg0.F, out_en0.F, out_pred0.F, out_backbone0.F, out_bottle0,\
                   out_seg1.F, out_en1.F, out_pred1.F, out_backbone1.F, out_bottle1
        else:
            # forward in backbone
            out_backbone, out_bottle= self.seg_model(x, is_seg=False)

            # forward in final
            out_seg = self.seg_model.final(out_backbone)

            return out_seg.F, out_backbone.F, out_bottle


class MinkUNet18_BYOL(nn.Module):

    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

        # define self-sup heads
        self.proj_dim = self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion

        self.relu = ME.MinkowskiReLU(inplace=True)

        # projector head
        self.encoder = nn.Sequential(
            ME.MinkowskiConvolution(
                self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim))

        # self.encoder = nn.Sequential(
        #     ME.MinkowskiConvolution(
        #         self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
        #         self.proj_dim,
        #         kernel_size=1,
        #         bias=True,
        #         dimension=self.seg_model.D),
        #     ME.MinkowskiBatchNorm(self.proj_dim),
        #     self.relu,
        #     ME.MinkowskiConvolution(
        #         self.proj_dim,
        #         self.seg_model.PLANES[7] * self.seg_model.BLOCK.expansion,
        #         kernel_size=1,
        #         bias=True,
        #         dimension=self.seg_model.D))

        # predictor head
        self.predictor = nn.Sequential(
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D),
            ME.MinkowskiBatchNorm(self.proj_dim),
            self.relu,
            ME.MinkowskiConvolution(
                self.proj_dim,
                self.proj_dim,
                kernel_size=1,
                bias=True,
                dimension=self.seg_model.D))

    def _forward_heads(self, x):
        out_seg = self.seg_model.final(x)
        out_en = self.encoder(x)
        out_pred = self.predictor(out_en)
        return out_seg, out_en, out_pred

    def forward(self, x, is_train=True, momentum=True):
        if is_train:
            if momentum:
                x0, _ = x
            else:
                _, x0 = x

            out_backbone0, out_bottle0 = self.seg_model(x0, is_seg=False)

            out_seg0, out_en0, out_pred0 = self._forward_heads(out_backbone0)

            return out_seg0.F, out_en0.F, out_pred0.F, out_backbone0.F, out_bottle0
        else:
            # forward in backbone
            out_backbone, out_bottle = self.seg_model(x, is_seg=False)

            # forward in final
            out_seg = self.seg_model.final(out_backbone)

            return out_seg.F, out_backbone.F, out_bottle


class MinkUNet18_MCMC(nn.Module):

    def __init__(self, seg_model, p_drop=0.5):
        super().__init__()
        self.seg_model = seg_model

        self.dropout = ME.MinkowskiDropout(p=p_drop)

    def forward(self, x, is_train=True):
        # forward in backbone
        out_backbone, out_bottle = self.seg_model(x, is_seg=False)
        out_backbone = self.dropout(out_backbone)
        # forward in final
        out_seg = self.seg_model.final(out_backbone)

        return out_seg.F, out_backbone.F, out_bottle

