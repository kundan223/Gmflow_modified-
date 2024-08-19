import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from backbone import CNNEncoder
from transformer_modified import FeatureTransformer, FeatureFlowAttention
from matching import global_correlation_softmax, local_correlation_softmax
from geometry import flow_warp
from utils import normalize_img, feature_add_position
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchsummary import summary


class GMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 **kwargs,
                 ):
        super(GMFlow, self).__init__()

        self.num_scales = num_scales = 1
        self.feature_channels = feature_channels = 128
        self.upsample_factor = upsample_factor = 8
        self.attention_type = attention_type = 'swin'
        self.num_transformer_layers = num_transformer_layers = 6

        # CNN backbone
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales)

        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              attention_type=attention_type,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        # flow propagation with self-attn
        self.feature_flow_attn = FeatureFlowAttention(in_channels=feature_channels)

        # convex upsampling: concat feature0 and flow as input
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)  # [2B, C, H, W]
        print(f"GMFlow: concat shape: {concat.shape}")
        features = self.backbone(concat)  # list of [2B, C, H, W], resolution from high to low
        print(f"GMFlow: features shape: {[feature.shape for feature in features]}")

        # reverse: resolution from low to high
        features = features[::-1]
        print(f"GMFlow: reversed features shape: {[feature.shape for feature in features]}")

        feature0, feature1 = [], []

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)  # tuple
            print(f"GMFlow: chunks shape: {[chunk.shape for chunk in chunks]}")
            feature0.append(chunks[0])
            feature1.append(chunks[1])

        print(f"GMFlow: feature0 shape: {[feature.shape for feature in feature0]}")
        print(f"GMFlow: feature1 shape: {[feature.shape for feature in feature1]}")

        return feature0, feature1

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=32,
                      ):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * upsample_factor
            print(f"GMFlow: up_flow (bilinear) shape: {up_flow.shape}")

        else:
            # convex upsampling
            concat = torch.cat((flow, feature), dim=1)
            print(f"GMFlow: concat shape: {concat.shape}")

            mask = self.upsampler(concat)
            print(f"GMFlow: mask shape: {mask.shape}")
            b, flow_channel, h, w = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
            print(f"GMFlow: reshaped mask shape: {mask.shape}")
            mask = torch.softmax(mask, dim=2)
            print(f"GMFlow: softmax mask shape: {mask.shape}")

            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            print(f"GMFlow: unfolded up_flow shape: {up_flow.shape}")
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]
            print(f"GMFlow: reshaped up_flow shape: {up_flow.shape}")

            up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
            print(f"GMFlow: summed up_flow shape: {up_flow.shape}")
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
            print(f"GMFlow: permuted up_flow shape: {up_flow.shape}")
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h,
                                      self.upsample_factor * w)  # [B, 2, K*H, K*W]
            print(f"GMFlow: final up_flow shape: {up_flow.shape}")

        return up_flow

    def forward(self, img0, img1,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                pred_bidir_flow=False,
                **kwargs,
                ):
        if attn_splits_list is None:
            attn_splits_list = [2] * self.num_scales
        if corr_radius_list is None:
            corr_radius_list = [-1] * self.num_scales
        if prop_radius_list is None:
            prop_radius_list = [-1] * self.num_scales
        results_dict = {}
        flow_preds = []

        img0, img1 = normalize_img(img0, img1)  # [B, 3, H, W]
        print(f"GMFlow: img0 shape: {img0.shape}")
        print(f"GMFlow: img1 shape: {img1.shape}")

        # resolution low to high
        feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features
        print(f"GMFlow: feature0_list shape: {[feature.shape for feature in feature0_list]}")
        print(f"GMFlow: feature1_list shape: {[feature.shape for feature in feature1_list]}")

        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx], feature1_list[scale_idx]
            print(f"GMFlow: feature0 shape at scale {scale_idx}: {feature0.shape}")
            print(f"GMFlow: feature1 shape at scale {scale_idx}: {feature1.shape}")

            if pred_bidir_flow and scale_idx > 0:
                # predicting bidirectional flow with refinement
                feature0, feature1 = torch.cat((feature0, feature1), dim=0), torch.cat((feature1, feature0), dim=0)
                print(f"GMFlow: concatenated feature0 shape at scale {scale_idx}: {feature0.shape}")
                print(f"GMFlow: concatenated feature1 shape at scale {scale_idx}: {feature1.shape}")

            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))
            print(f"GMFlow: upsample_factor at scale {scale_idx}: {upsample_factor}")

            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
                print(f"GMFlow: upsampled flow shape at scale {scale_idx}: {flow.shape}")

            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                print(f"GMFlow: warped feature1 shape at scale {scale_idx}: {feature1.shape}")

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]
            print(f"GMFlow: attn_splits at scale {scale_idx}: {attn_splits}")
            print(f"GMFlow: corr_radius at scale {scale_idx}: {corr_radius}")
            print(f"GMFlow: prop_radius at scale {scale_idx}: {prop_radius}")

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            print(f"GMFlow: feature0 shape after adding position at scale {scale_idx}: {feature0.shape}")
            print(f"GMFlow: feature1 shape after adding position at scale {scale_idx}: {feature1.shape}")

            # Transformer
            feature0, feature1 = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            print(f"GMFlow: feature0 shape after transformer at scale {scale_idx}: {feature0.shape}")
            print(f"GMFlow: feature1 shape after transformer at scale {scale_idx}: {feature1.shape}")

            # correlation and softmax
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax(feature0, feature1, pred_bidir_flow)[0]
            else:  # local matching
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
            print(f"GMFlow: flow_pred shape at scale {scale_idx}: {flow_pred.shape}")

            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred
            print(f"GMFlow: flow shape at scale {scale_idx}: {flow.shape}")

            # upsample to the original resolution for supervision
            if self.training:  # only need to upsample intermediate flow predictions at training time
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                print(f"GMFlow: flow_bilinear shape at scale {scale_idx}: {flow_bilinear.shape}")
                flow_preds.append(flow_bilinear)

            # flow propagation with self-attn
            if pred_bidir_flow and scale_idx == 0:
                feature0 = torch.cat((feature0, feature1), dim=0)  # [2*B, C, H, W] for propagation
                print(f"GMFlow: concatenated feature0 shape for propagation at scale {scale_idx}: {feature0.shape}")
            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius)
            print(f"GMFlow: flow shape after feature_flow_attn at scale {scale_idx}: {flow.shape}")

            # bilinear upsampling at training time except the last one
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                print(f"GMFlow: flow_up shape at scale {scale_idx}: {flow_up.shape}")
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                print(f"GMFlow: final flow_up shape: {flow_up.shape}")
                flow_preds.append(flow_up)

        results_dict.update({'flow_preds': flow_preds})
        print(f"GMFlow: flow_preds shape: {[flow_pred.shape for flow_pred in flow_preds]}")

        return results_dict