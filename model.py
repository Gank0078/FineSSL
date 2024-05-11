import math
from operator import mul
from functools import reduce
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

import copy


class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)
        
    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class Adapter(nn.Module):
    def __init__(self, in_dim, bottle_dim, adapter_scalar=0.1, scalar_learnable=False, adapter_type='parallel', dtype=None):
        super().__init__()
        self.adapter_type = adapter_type

        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)

        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

        # self.scale = float(adapter_scalar)
        if scalar_learnable is True:
            # self.scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones([]) * adapter_scalar)
        else:
            self.scale = float(adapter_scalar)

    def forward(self, x):
        out = self.ln(x)
        out = self.down_proj(out)
        out = self.relu(out)
        out = self.up_proj(out)

        out = out * self.scale

        if self.adapter_type == 'sequential':
            return out
        elif self.adapter_type == 'parallel':
            return out + x


class AdaptFormer(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=64,
                 dropout=0.1,
                 # init_option="bert",
                 init_option="lora",
                 adapter_scalar=0.1,
                 scalar_learnable=False,
                 adapter_layernorm_option="in", dtype=None):
        super().__init__()
        # self.n_embd = config.d_model if d_model is None else d_model
        # self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd, dtype=dtype)

        if scalar_learnable is True:
            # self.scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones([]) * adapter_scalar)
        else:
            self.scale = float(adapter_scalar)

        # self.scale = nn.Parameter(torch.ones(1)).to(dtype)

        self.down_proj = nn.Linear(self.n_embd, self.down_size, dtype=dtype)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd, dtype=dtype)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class LoRA(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(in_dim, bottle_dim, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(bottle_dim, in_dim, dtype=dtype))
        self.scaling = 1.0 / bottle_dim
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        x = x @ self.lora_A
        x = x @ self.lora_B
        x = self.scaling * x
        return x


class SSF(nn.Module):
    def __init__(self, in_dim, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(in_dim, dtype=dtype))
        nn.init.normal_(self.scale, mean=1.0, std=0.02)
        nn.init.normal_(self.shift, std=0.02)

    def forward(self, x):
        return x * self.scale + self.shift


class multi_SSF(nn.Module):
    def __init__(self, n_cls, in_dim, init_mean=1.0, init_std=0.02, dtype=None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_cls, in_dim, dtype=dtype))
        self.shift = nn.Parameter(torch.zeros(n_cls, in_dim, dtype=dtype))
        # nn.init.normal_(self.scale, mean=1.0, std=0.02)
        # nn.init.normal_(self.shift, std=0.02)
        nn.init.normal_(self.scale, mean=init_mean, std=init_std)
        nn.init.normal_(self.shift, std=init_std)

    def forward(self, x, norm=False):
        if norm is True:
            x = x / x.norm(dim=-1, keepdim=True)
        return x * self.scale + self.shift


class ViT_Head(nn.Module):
    # def __init__(self, text_features, visual_proj, logit_scale, rand_init=False):
    def __init__(self, num_classes, visual_proj, emb_dim, text_features=None):
        super().__init__()
        # in_dim = visual_proj.shape[0]
        # n_cls = text_features.shape[0]
        n_cls = num_classes

        self.weight = nn.Parameter(torch.empty(n_cls, emb_dim))

        if text_features is not None:
            self.weight.data = text_features.data @ visual_proj.data.t()
        else:
            self.weight.data.uniform_(-1, 1).renorm_(2, 0, 1e-5).mul_(1e5)

        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class ViT_Head_v0(nn.Module):
    def __init__(self, text_features, visual_proj, logit_scale):
        super().__init__()
        self.visual_proj = nn.Parameter(visual_proj.clone())
        self.weight = nn.Parameter(text_features.clone())
        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

    def forward(self, x):
        x = x @ self.visual_proj
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class ViT_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        n_layers = clip_model.visual.transformer.layers
        emb_dim = clip_model.visual.transformer.width
        seq_len = clip_model.visual.positional_embedding.shape[0]
        patch_size = clip_model.visual.conv1.kernel_size
        dtype = clip_model.dtype

        use_finetune = cfg.finetune
        use_bias_tuning = cfg.bias_tuning
        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_vpt_last = cfg.vpt_last

        use_adapter = cfg.adapter
        use_lora = cfg.lora
        use_ssf = cfg.ssf
        vpt_len = cfg.vpt_len
        adapter_dim = cfg.adapter_dim
        lora_dim = cfg.lora_dim
        partial = cfg.partial
        block_num = cfg.block_num

        use_adaptformer = cfg.adaptformer
        ffn_num = cfg.ffn_num

        if partial is None:
            partial = n_layers
        else:
            partial = int(partial)

        block_list = []
        if block_num is not None:
            block_num = int(block_num)
            init_block = n_layers - 1
            for i in range(block_num):
                block_list.append(init_block - i)

        blocks = clip_model.visual.transformer.resblocks

        if use_finetune:
            if block_num is None:
                finetune_list = nn.ParameterList([
                    param for name, param in clip_model.visual.named_parameters()
                ])
            else:
                finetune_list = nn.ParameterList([])
                for name, param in clip_model.visual.named_parameters():
                    for num in block_list:
                        if 'resblocks.' + str(num) in name:
                            finetune_list.append(param)
                            continue
        else:
            finetune_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[None] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        elif use_vpt_last:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - 1),
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype)
            ])
        else:
            vpt_list = [None] * n_layers

        # if use_adapter:
        #     adapter_list = nn.ModuleList([
        #         *[None] * (n_layers - partial),
        #         *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype) for _ in range(partial)]
        #     ])
        # else:
        #     adapter_list = [None] * n_layers

        if use_adapter:
            adapter_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[Adapter(in_dim=emb_dim, bottle_dim=adapter_dim, adapter_scalar=cfg.adapter_scalar,
                          scalar_learnable=cfg.scalar_learnable, dtype=dtype) for _ in range(partial)]
            ])
        else:
            adapter_list = [None] * n_layers

        if use_lora:
            lora_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "q": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                    "v": LoRA(in_dim=emb_dim, bottle_dim=lora_dim, dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            lora_list = [None] * n_layers

        if use_ssf:
            _block_0 = clip_model.visual.transformer.resblocks[0]
            ssf_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[nn.ModuleDict({
                    "attn_in": SSF(_block_0.attn.in_proj_bias.shape[0], dtype=dtype),
                    "attn_out": SSF(_block_0.attn.out_proj.bias.shape[0], dtype=dtype),
                    "mlp_in": SSF(_block_0.mlp[0].bias.shape[0], dtype=dtype),
                    "mlp_out": SSF(_block_0.mlp[2].bias.shape[0], dtype=dtype),
                }) for _ in range(partial)]
            ])
        else:
            ssf_list = [None] * n_layers

        if use_adaptformer:
            adaptformer_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[AdaptFormer(d_model=emb_dim, bottleneck=ffn_num, adapter_scalar=cfg.adapter_scalar, scalar_learnable=cfg.scalar_learnable,
                              adapter_layernorm_option=cfg.ln_opt, dtype=dtype) for _ in
                  range(partial)]
            ])
        else:
            adaptformer_list = [None] * n_layers

        visual_proj = clip_model.visual.proj.data
        logit_scale = clip_model.logit_scale.data

        # head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)

        if cfg.rand_init:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
        else:
            head = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)

        if cfg.rand_init1:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768).to(clip_model.dtype)
        else:
            head1 = ViT_Head(cfg.DATA.NUMBER_CLASSES, visual_proj, 768, text_features=text_features).to(clip_model.dtype)

        # To be optimized
        self.finetune_list = finetune_list
        self.bias_list = bias_list
        self.vpt_list = vpt_list
        self.adapter_list = adapter_list
        self.lora_list = lora_list
        self.ssf_list = ssf_list
        self.head = head
        self.head1 = head1

        self.adaptformer_list = adaptformer_list
        self.ffn_opt = cfg.ffn_opt

        self.proj = copy.deepcopy(clip_model.visual.proj)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

        self.alpha_mat = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, cfg.DATA.NUMBER_CLASSES), dtype=dtype) * cfg.alpha)
        self.text_emb = nn.Parameter(text_features.clone())

        self.alpha_cls = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, ), dtype=dtype) * cfg.alpha)
        self.alpha = nn.Parameter(torch.ones([]) * cfg.alpha)


class CLIP_ViT(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        self.dtype = clip_model.dtype

    def forward(self, x, tuner=None):
        x = x.to(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        _bsz = x.shape[0]
        _seq_len = x.shape[1]
        _emb_dim = x.shape[2]

        n_layers = self.transformer.layers

        for i in range(n_layers):
            block = self.transformer.resblocks[i]

            if tuner is not None:
                vpt = tuner.vpt_list[i]
                adapter = tuner.adapter_list[i]
                lora = tuner.lora_list[i]
                ssf = tuner.ssf_list[i]
                adaptformer = tuner.adaptformer_list[i]
            else:
                vpt = adapter = lora = ssf = adaptformer = None

            if vpt is not None:
                x = vpt(x)

            _seq_len_after_vpt = x.shape[1]

            x = x.permute(1, 0, 2)  # NLD -> LND

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            _attn_in_proj_weight = _attn.in_proj_weight
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj = _mlp[0]
            _mlp_gelu = _mlp[1]
            _mlp_out_proj = _mlp[2]

            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            residual = x  # deep copy

            x = _ln_1(x)

            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias)
            if ssf is not None:
                qkv = ssf["attn_in"](qkv)
            q, k, v = qkv.chunk(3, dim=-1)

            if lora is not None:
                q = q + lora["q"](x)
                v = v + lora["v"](x)

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            q = q / math.sqrt(_head_dim)
            attn = torch.bmm(q, k.transpose(-2, -1))
            attn = F.softmax(attn, dim=-1)
            x = torch.bmm(attn, v)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            
            x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
            if ssf is not None:
                x = ssf["attn_out"](x)
            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)

            x = residual + x

            if adaptformer is not None:
                adapt_x = adaptformer(x, add_residual=False)

            ##########################
            ## Feed-Forward Network ##
            ##########################
            residual = x  # deep copy

            x = _ln_2(x)

            x = _mlp_in_proj(x)
            if ssf is not None:
                x = ssf["mlp_in"](x)
            x = _mlp_gelu(x)
            x = _mlp_out_proj(x)
            if ssf is not None:
                x = ssf["mlp_out"](x)
            
            if adapter is not None:
                x = adapter(x)

            if adaptformer is not None:
                if tuner.ffn_opt == "parallel":
                    x = x + adapt_x
                elif tuner.ffn_opt == "sequential":
                    x = adaptformer(x)
            
            x = residual + x
            
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])
        return x


class RN_Head(nn.Module):
    def __init__(self, text_features, logit_scale):
        super().__init__()
        n_cls = text_features.shape[0]
        self.weight = nn.Parameter(text_features.clone())
        # self.logit_scale = nn.Parameter(logit_scale.exp().clone())
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))
        # self.logit_scale = torch.ones([]) * (1 / 0.05)

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        weight = self.weight / self.weight.norm(dim=-1, keepdim=True)
        return self.logit_scale * (x @ weight.t())


class RN_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        out_dim = clip_model.visual.output_dim
        dtype = clip_model.dtype

        use_finetune = cfg.finetune
        use_bias_tuning = cfg.bias_tuning
        use_bn_tuning = cfg.bn_tuning
        use_ssf = cfg.ssf

        blocks = nn.Sequential(*[
            *clip_model.visual.layer1,
            *clip_model.visual.layer2,
            *clip_model.visual.layer3,
            *clip_model.visual.layer4,
        ])

        if use_finetune:
            finetune_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
            ])
        else:
            finetune_list = None

        if use_bias_tuning:
            bias_list = nn.ParameterList([
                param for name, param in clip_model.visual.named_parameters()
                if name.endswith("bias")
            ])
        else:
            bias_list = None

        if use_bn_tuning:
            bn_list = nn.ModuleList([
                clip_model.visual.bn1,
                clip_model.visual.bn2,
                clip_model.visual.bn3,
                *[nn.ModuleList([
                    block.bn1,
                    block.bn2,
                    block.bn3,
                ]) for block in blocks],
            ])
        else:
            bn_list = None

        if use_ssf:
            ssf_list = nn.ModuleList([
                SSF(out_dim, dtype=dtype),
            ])
        else:
            ssf_list = None

        logit_scale = clip_model.logit_scale.data
        head = RN_Head(text_features, logit_scale).to(clip_model.dtype)

        # To be optimized
        self.finetune_list = finetune_list
        self.bias_list = bias_list
        self.bn_list = bn_list
        self.ssf_list = ssf_list
        self.head = head

class CLIP_RN(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.conv1 = clip_model.visual.conv1
        self.bn1 = clip_model.visual.bn1
        self.conv2 = clip_model.visual.conv2
        self.bn2 = clip_model.visual.bn2
        self.conv3 = clip_model.visual.conv3
        self.bn3 = clip_model.visual.bn3
        self.avgpool = clip_model.visual.avgpool
        self.relu = clip_model.visual.relu
        self.layer1 = clip_model.visual.layer1
        self.layer2 = clip_model.visual.layer2
        self.layer3 = clip_model.visual.layer3
        self.layer4 = clip_model.visual.layer4
        self.attnpool = clip_model.visual.attnpool
        self.dtype = clip_model.dtype
    
    def forward(self, x, tuner=None):
        
        x = x.to(self.dtype)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        ssf_list = tuner.ssf_list
        if ssf_list is not None:
            x = ssf_list[0](x)
        
        return x


class Model(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        if cfg.backbone.startswith("ViT"):
            self.image_encoder = CLIP_ViT(clip_model)
            self.tuner = ViT_Tuner(cfg, clip_model, text_features)
        else:
            self.image_encoder = CLIP_RN(clip_model)
            self.tuner = RN_Tuner(cfg, clip_model)

    def forward(self, image):
        feat = self.image_encoder(image, self.tuner)
        return feat
