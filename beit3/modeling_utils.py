# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

import os
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12, 
        checkpoint_activations=checkpoint_activations, 
    )


def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16, 
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24, 
        checkpoint_activations=checkpoint_activations, 
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        # 1. Inizializzazione del modello BEiT-3 standard
        self.beit3 = BEiT3(args)
        # 2. Logica di caricamento condizionale della proiezione multilingue
        if hasattr(args, 'xlmr_emb_path') and args.xlmr_emb_path:
            self._load_multilingual_embeddings(args.xlmr_emb_path)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# 3. NUOVO METODO: Caricamento e Sostituzione degli Embedding
    def _load_multilingual_embeddings(self, emb_path):
        """Sostituisce gli embedding testuali di BEiT-3 con la matrice XLM-R proiettata."""
        
        # Stampa di debug per tracciare l'operazione
        print(f"Loading and replacing text embeddings from {emb_path} for multilingual projection.")
        
        # Carica la matrice di embedding proiettata (su CPU per sicurezza)
        try:
            new_emb_weight = torch.load(emb_path, map_location='cpu') 
        except FileNotFoundError:
            raise FileNotFoundError(f"Multilingual embedding file not found at: {emb_path}")

        V, D = new_emb_weight.shape
        
        # Verifica che la dimensione dell'embedding (D) sia compatibile con l'encoder
        expected_dim = self.args.encoder_embed_dim
        if D != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: loaded {D} but expected {expected_dim}. Check your projection script.")
        
        # 1. Crea un nuovo layer nn.Embedding
        new_text_embed = nn.Embedding(V, D)
        
        # 2. Carica i pesi proiettati nel nuovo layer
        new_text_embed.weight.data.copy_(new_emb_weight)
        
        # 3. Sostituzione del layer originale
        # Questo sovrascrive il modulo 'text_embed' all'interno di BEiT3 (torchscale)
        self.beit3.text_embed = new_text_embed
        
        # 4. Aggiorna la dimensione del vocabolario negli argomenti
        self.args.vocab_size = V
        
        print(f"Successfully replaced text embeddings. New vocab size: {V}.")
        
    def get_input_embeddings(self) -> nn.Embedding:
        return self.beit3.text_embed
            
