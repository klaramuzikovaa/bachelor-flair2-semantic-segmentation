import torch
import torch.nn as nn
import segmentation_models_pytorch as smp  
from src.backbones.utae_model import UTAE
from src.backbones.fusion_utils import *

class TimeTexture_flair(nn.Module):
    def __init__(self, config):
        super(TimeTexture_flair, self).__init__()  
        self.config = config
        self.encoder_name = config.get("encoder_name", "resnet34")
        aerial_channels = config['num_channels_aerial']

        self.adapter = nn.Sequential(
            nn.Conv2d(aerial_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3)
        )
        
        self.arch_vhr = smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights="imagenet",
            in_channels=3, 
            classes=config['num_classes']
        )
        
        self.arch_sen = UTAE(
            input_dim=config['num_channels_sat'],
            encoder_widths=config['encoder_widths'], 
            decoder_widths=config['decoder_widths'],
            out_conv=config['out_conv'],
            str_conv_k=config['str_conv_k'],
            str_conv_s=config['str_conv_s'],
            str_conv_p=config['str_conv_p'],
            agg_mode=config['agg_mode'], 
            encoder_norm=config['encoder_norm'],
            n_head=config['n_head'], 
            d_model=config['d_model'], 
            d_k=config['d_k']
        )

        # --- KLÍČOVÁ OPRAVA ---
        # Tento adaptér převede 13 kanálů ze satelitu na 256 kanálů pro ASPP
        self.sat_to_aspp_adapter = nn.Conv2d(config['num_classes'], 256, kernel_size=1)
        # ----------------------

        self.fusion_weight = nn.Parameter(torch.zeros(1))

    def forward(self, config, aerial, satellite, dates, metadata, *args):
        vhr_aerial = self.adapter(aerial)
        features = self.arch_vhr.encoder(vhr_aerial)
    
        utae_fmaps_raw = self.arch_sen(satellite, dates)
        utae_logits = utae_fmaps_raw[-1] if isinstance(utae_fmaps_raw, list) else utae_fmaps_raw
        utae_embedding = torch.sigmoid(utae_logits)

        # Dekodér DeepLabV3+
        aspp_features = self.arch_vhr.decoder.aspp(features[-1])
        
        # Fúze se satelitem
        utae_fused = torch.nn.functional.interpolate(
            utae_embedding, size=aspp_features.shape[2:], mode='bilinear', align_corners=False
        )
        
        # --- KLÍČOVÁ OPRAVA ---
        # Tady použijeme adaptér, aby se kanály shodovaly (13 -> 256)
        utae_fused_projected = self.sat_to_aspp_adapter(utae_fused)
        aspp_features = aspp_features + self.fusion_weight * utae_fused_projected
        # ----------------------
        
        aspp_features = self.arch_vhr.decoder.up(aspp_features)
        high_res_features = self.arch_vhr.decoder.block1(features[2])
        
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.arch_vhr.decoder.block2(concat_features)
        
        output_unet = self.arch_vhr.segmentation_head(fused_features)
        
        utae_logits_upsampled = torch.nn.functional.interpolate(
            utae_logits, size=(512, 512), mode='bilinear', align_corners=False
        )
    
        return utae_logits_upsampled, output_unet