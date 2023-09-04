import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 num_combine_layers=4,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 is_res=False,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)
        
        if is_res:
            combine_net = []
            for l in range(num_layers):
                if l == 0:
                    in_dim = self.in_dim 
                else:
                    in_dim = hidden_dim
                
                if l == num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = hidden_dim
                
                combine_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.combine_net = nn.ModuleList(combine_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None


    def forward(self, x, d,bg_model=None,return_features=False,return_bg_raw=False, return_mixnet=False):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # sigma
        
               
        bg_raw_sigma = None 
        bg_raw_color = None
        bg_sigma = None 
        bg_color = None
        
        if bg_model is not None:
            bg_sigma, bg_color, bg_raw_sigma, bg_raw_color = bg_model(x,d,return_features=True)

        x = self.encoder(x, bound=self.bound)

        combine_param_res = None
        if bg_sigma is not None:
            # bg_sigma_expanded = bg_raw_sigma.unsqueeze(-1)
            # h = torch.cat([x, bg_sigma_expanded], dim=-1)
            h = x
            for l in range(self.num_layers):
                h = self.combine_net[l](h)
                if l != self.num_layers - 1:
                    h = F.relu(h, inplace=True)
        
            h_sigmoided = torch.sigmoid(h[..., 0])
            combine_param_res = h_sigmoided
            combine_param_bg = (1.0-h_sigmoided)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # raw_sigma = h[..., 0].clone()
        
        if combine_param_res is None:
            combine_param_res = torch.ones_like(h[..., 0])
       

        if bg_raw_sigma is not None:
            
            # sigma_mask = bg_raw_sigma > bg_thresh
            # combine_param_res[sigma_mask] = 0.0
            # combine_param_bg = (1.0-combine_param_res)

            # use combine_param to weigh between bg and fg
            raw_sigma_combined = h[...,0] * combine_param_res + bg_raw_sigma * combine_param_bg
            
            # raw_sigma_combined = h[..., 0] + bg_raw_sigma
            # raw_sigma_combined = h[...,0].clone()
            # raw_sigma_combined[sigma_mask] = bg_raw_sigma[sigma_mask]
            # h = bg_raw_sigma
        else:
            raw_sigma_combined = h[...,0]

        raw_sigma = raw_sigma_combined.clone()
        # raw_sigma_combined = h[...,0]
        sigma = trunc_exp(raw_sigma_combined)
        
        # if bg_sigma is not None:
        #     sigma = sigma + bg_sigma
            
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        raw_color = h.clone()
        # sigmoid activation for rgb
        # combine_param = combine_param.unsqueeze(-1).expand_as(h)
        
        
        if bg_raw_color is not None:
            combine_param_res_expanded = torch.stack([combine_param_res,combine_param_res,combine_param_res],dim=-1)
            combine_param_bg_expanded = torch.stack([combine_param_bg,combine_param_bg,combine_param_bg],dim=-1)
            rgb_raw = h * combine_param_res_expanded + bg_raw_color * combine_param_bg_expanded
            # rgb_raw = h.clone()
            # rgb_raw[sigma_mask] = bg_raw_color[sigma_mask]
            # rgb_raw = h + bg_raw_color
        else:
            rgb_raw = h
            # h: [N, 3]
            # h = h*
            
            # h = h + bg_raw_color
            # h = bg_raw_color
        # rgb_raw = h
        color = torch.sigmoid(rgb_raw)

        result = (sigma, color)
        if return_features:
            result += (raw_sigma, raw_color)
        
        if return_bg_raw:
            result += (bg_sigma,bg_color)
        
        if return_mixnet:
            result += (combine_param_res,)

                
        
        return result

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
