"""HF (Qwen3) modeline Engram enjeksiyonunu hook'larla takma / cikarma.

Tasarim:
- Her hedef katmana forward-pre-hook baglanir; katmanin GIRIS hidden state'i
  `h + alpha * engram(h, input_ids)` ile degistirilir.
- input_ids, base modelin ust seviye forward'ini saran bir context ile aktarilir.
- detach() ile tamamen cikarilabilir -> model birebir orijine doner.
"""

from typing import List

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .hashing import HashConfig
from .module import EngramInjection, EngramModuleConfig


class EngramAttach:
    """Bir HF causal LM'e birden fazla EngramInjection modulu takar.

    Ornek:
        attach = EngramAttach(model, hash_cfg, module_cfg)
        with attach.enabled():          # takili calisir
            out = model(**inputs)
        attach.remove()                 # tamamen soke et
    """

    def __init__(
        self,
        model: PreTrainedModel,
        hash_cfg: HashConfig,
        module_cfg: EngramModuleConfig,
        layer_ids: List[int] = None,
    ):
        self.model = model
        self.layer_ids = layer_ids if layer_ids is not None else list(hash_cfg.layer_ids)
        self.hash_cfg = hash_cfg

        base = model.model  # Qwen3ForCausalLM -> Qwen3Model
        layers = base.layers
        self.hidden_size = base.config.hidden_size

        self.injections = nn.ModuleDict()
        shared_embedding = None
        for lid in self.layer_ids:
            inj = EngramInjection(
                hash_cfg, module_cfg, self.hidden_size, shared_embedding=shared_embedding
            )
            if shared_embedding is None:
                shared_embedding = inj.multi_head_embedding  # ilk katmanin tablosu ortak
            self.injections[str(lid)] = inj

        # Cihaza tasir ama DTYPE'A CAST ETMEYIZ: egitilebilir moduller fp32 kalmali.
        # bf16 parametrelerde lr-scale guncellemeler yuvarlanip kayboluyor (C-v2 bug'i).
        device = next(model.parameters()).device
        self.injections.to(device=device)

        self._current_input_ids = None
        self._hooks = []

        # Ust seviye forward'da input_ids yakala
        self._hooks.append(
            base.register_forward_pre_hook(self._capture_input_ids, with_kwargs=True)
        )
        # Hedef katmanlarin girisini degistir
        for lid in self.layer_ids:
            layer_module = layers[lid]
            inj = self.injections[str(lid)]
            self._hooks.append(
                layer_module.register_forward_pre_hook(
                    self._make_layer_hook(inj, lid), with_kwargs=True
                )
            )

    def _capture_input_ids(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self._current_input_ids = ids

    def _make_layer_hook(self, injection: EngramInjection, layer_id: int):
        def hook(module, args, kwargs):
            if not self.enabled_flag or self._current_input_ids is None:
                return None
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None:
                return None
            delta = injection(hidden, self._current_input_ids, layer_id)
            new_hidden = hidden + delta
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = new_hidden
                return args, kwargs
            return (new_hidden,) + tuple(args[1:]), kwargs

        return hook

    @property
    def enabled_flag(self) -> bool:
        return getattr(self, "_enabled", True)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def trainable_parameters(self):
        """Sadece Engram modullerinin parametreleri (backbone donuk).

        Katmanlar arasi paylasilan embedding tablosu TEK KEZ dondurulur
        (yoksa optimizer ayni tensor'u birden fazla kez gunceller).
        """
        seen = set()
        params = []
        for inj in self.injections.values():
            for p in inj.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    def mark_only_engram_trainable(self):
        """Backbone'un tum parametrelerini dondurur."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        for inj in self.injections.values():
            for p in inj.parameters():
                p.requires_grad_(True)
        # Embedding + lm_head tie edilmişse tekrar dondurulmus olur; kontrol:
        tied = getattr(self.model.config, "tie_word_embeddings", False)
        if tied and hasattr(self.model.get_output_embeddings(), "weight"):
            out_w = self.model.get_output_embeddings().weight
            in_w = self.model.get_input_embeddings().weight
            if out_w is in_w:
                out_w.requires_grad_(False)

    def remove(self):
        """Tum hook'lari ve modulleri tamamen soker -> model birebir orijinal."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.injections = None

    def alpha_values(self):
        """Katman bazinda dis gate degerleri (analiz icin)."""
        return {lid: float(inj.alpha.item()) for lid, inj in self.injections.items()}
