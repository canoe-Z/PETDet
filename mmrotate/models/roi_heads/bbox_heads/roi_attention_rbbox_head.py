import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ...builder import ROTATED_HEADS
from .convfc_rbbox_head import RotatedConvFCBBoxHead
import math


@ROTATED_HEADS.register_module()
class RoIAttentionBBoxHead(RotatedConvFCBBoxHead):

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 attention_hidden_channels=128,
                 attention_pool_size=2,
                 attention_pool_size_gram=2,
                 subsample='naive',
                 combination='cas_rram_gram',
                 *args,
                 **kwargs):
        super().__init__(num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, num_reg_convs, num_reg_fcs,
                         conv_out_channels, fc_out_channels, conv_cfg, norm_cfg, *args, **kwargs)
        self.attention_hidden_channels = attention_hidden_channels
        self.conv_out_channels = conv_out_channels
        self.attention_pool_size = attention_pool_size
        self.attention_pool_size_gram = attention_pool_size_gram
        self.subsample = subsample
        self.combination = combination
        
        if 'rram' in combination:
            self.q_conv_rram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)

            if subsample == 'naive':
                self.k_conv_rram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)
                self.v_conv_rram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size)

            elif subsample == 'maxpool':
                self.k_conv_rram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
                self.v_conv_rram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)

            self.y_conv_rram = nn.Conv2d(attention_hidden_channels, conv_out_channels, 1)
          
        if 'gram' in combination:
            self.q_conv_gram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
            
            if subsample == 'naive': 
                self.k_conv_gram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size_gram)
                self.v_conv_gram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1, stride=attention_pool_size_gram)
            
            elif subsample == 'maxpool':
                self.k_conv_gram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)
                self.v_conv_gram = nn.Conv2d(conv_out_channels, attention_hidden_channels, 1)

            self.y_conv_gram = nn.Conv2d(attention_hidden_channels, conv_out_channels, 1)
            
        
    def init_weights(self):
        super(RoIAttentionBBoxHead, self).init_weights()
        # conv layers are already initialized by ConvModule
        
    def gram(self, roi_feats, feature):
        BS, C, H, W = feature.shape
        BS_num_rois, C_roi, roi_h, roi_w = roi_feats.shape
        num_rois = BS_num_rois // BS
        Q = self.q_conv_gram(roi_feats)  # (BS*num_rois, attention_hidden_channels, H, W)
        
        
        if self.subsample == 'maxpool':
            feature = F.max_pool2d(feature, self.attention_pool_size, self.attention_pool_size)

        # stride=2
        #_x = F.max_pool2d(feature, self.attention_pool_size, self.attention_pool_size)
        #_x = self.ds_conv_g(feature)
        #_H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        
        _H, _W = math.ceil(H / self.attention_pool_size_gram), math.ceil(W / self.attention_pool_size_gram)
        K = self.k_conv_gram(feature)
        V = self.v_conv_gram(feature)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*roi_h*roi_w, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, _H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, _H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, roi_h, roi_w, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        Y = Y.contiguous()
        y = self.y_conv_gram(Y)
        y = y.contiguous()

        return y
        
        
        
    def rram(self, roi_feats, BS, num_rois):
        BS_num_rois, C, H, W = roi_feats.shape
        Q = self.q_conv_rram(roi_feats)  # (BS*num_rois, attention_hidden_channels, H, W)
        _H, _W = math.ceil(H / self.attention_pool_size), math.ceil(W / self.attention_pool_size)
        #_H, _W = H // self.attention_pool_size, W // self.attention_pool_size
        if self.subsample == 'maxpool':
            roi_feats = F.max_pool2d(roi_feats, self.attention_pool_size, self.attention_pool_size, ceil_mode=True)
        #_x = self.ds_conv(x)
        #layer_norm = nn.LayerNorm([self.conv_out_channels, _H, _W])
        #_x = layer_norm(_x)
        K = self.k_conv_rram(roi_feats)  # (BS*num_rois, attention_hidden_channels, _H, _W)
        V = self.v_conv_rram(roi_feats)  # (BS*num_rois, attention_hidden_channels, _H, _W)

        Q = Q.permute(0, 2, 3, 1)  # (BS*num_rois, H, W, attention_hidden_channels)
        Q = Q.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Q = Q.reshape(BS, num_rois*H*W, self.attention_hidden_channels)  # (BS, num_rois*H*W, attention_hidden_channels)

        K = K.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        K = K.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        K = K.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        V = V.permute(0, 2, 3, 1)  # (BS*num_rois, _H, _W, attention_hidden_channels)
        V = V.reshape(BS, num_rois, _H, _W, self.attention_hidden_channels)
        V = V.reshape(BS, num_rois*_H*_W, self.attention_hidden_channels)  # (BS, num_rois*_H*_W, attention_hidden_channels)

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        #print_shape(Q,K,V)
        
        WEIGHTS = torch.bmm(Q, K.permute(0, 2, 1))  # (BS, num_rois*H*W, num_rois*_H*_W)
        WEIGHTS = torch.softmax(WEIGHTS, dim=2)

        Y = torch.bmm(WEIGHTS, V)  # (BS, num_rois*H*W, attention_hidden_channels)
        Y = Y.reshape(BS, num_rois, H, W, self.attention_hidden_channels)
        Y = Y.reshape(BS*num_rois, H, W, self.attention_hidden_channels)
        Y = Y.permute(0, 3, 1, 2)
        Y = Y.contiguous()
        y = self.y_conv_rram(Y)
        y = y.contiguous()
        
        return y
    
    
    def forward(self, x, feats):
        """

        :param x: shape (BS, num_rois, C, H, W)
        :return:
        """
        # print(x.shape)
        # assert(1==2)
        BS, num_rois, C, H, W = x.shape
        x = x.reshape(BS*num_rois, C, H, W)
        if self.combination=='rram':
            x = self.rram(x, BS, num_rois) + x
            
        elif self.combination=='gram':
            x = self.gram(x, feats) + x
            
        elif self.combination=='cas_rram_gram':
            x = self.rram(x, BS, num_rois) + x
            x = self.gram(x, feats) + x
        
        elif self.combination=='cas_gram_rram':
            x = self.gram(x, feats) + x
            x = self.rram(x, BS, num_rois) + x
            
        elif self.combination=='par_rram_gram':
            z1 = self.rram(x, BS, num_rois)
            z2 = self.gram(x, feats)
            x = z1 + z2 + x   

        return super(RoIAttentionBBoxHead, self).forward(x)


@ROTATED_HEADS.register_module()
class Shared2FCRoIAttentionBBoxHead(RoIAttentionBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCRoIAttentionBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@ROTATED_HEADS.register_module()
class UnsharedConvFCRoIAttentionBBoxHead(RoIAttentionBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(UnsharedConvFCRoIAttentionBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=0,
            num_cls_convs=0,
            num_cls_fcs=2,
            num_reg_convs=2,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)