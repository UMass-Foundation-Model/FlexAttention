#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def set_hd_info(self, hd_features, image_range_list, inputs_embeds, step, ori_sizes):
        if hd_features is not None:
            cos, sin = self.model.layers[0].self_attn.rotary_emb(inputs_embeds, seq_len=inputs_embeds.shape[1])
            image_begin_id = None
            image_end_id = None
            for image_range in image_range_list:
                if len(image_range) != 0:
                    image_begin_id = image_range[0][0]
                    image_end_id = image_range[0][1]
                    break
            if image_end_id is not None:
                size = int((image_end_id - image_begin_id)**0.5)
                image_cos = cos[image_begin_id: image_end_id].reshape(size, size, -1).unsqueeze(0).permute(0, 3, 1, 2)
                image_sin = sin[image_begin_id: image_end_id].reshape(size, size, -1).unsqueeze(0).permute(0, 3, 1, 2)
                hd_size = int(hd_features.shape[1]**0.5)
                image_cos = torch.nn.functional.interpolate(image_cos, size=hd_size, mode="bicubic")
                image_sin = torch.nn.functional.interpolate(image_sin, size=hd_size, mode="bicubic")
                for i in range(len(self.model.layers)):
                    self.model.layers[i].self_attn.image_cos = image_cos.flatten(2).permute(0, 2, 1).unsqueeze(0)
                    self.model.layers[i].self_attn.image_sin = image_sin.flatten(2).permute(0, 2, 1).unsqueeze(0)
                    self.model.layers[i].self_attn.hd_features = hd_features
                    self.model.layers[i].self_attn.image_range_list = image_range_list
                    self.model.layers[i].self_attn.step = step
                    self.model.layers[i].self_attn.ori_sizes = ori_sizes
            else:
                for i in range(len(self.model.layers)):
                    self.model.layers[i].self_attn.image_cos = 0
                    self.model.layers[i].self_attn.image_sin = 0
                    self.model.layers[i].self_attn.hd_features = hd_features
                    self.model.layers[i].self_attn.image_range_list = image_range_list
                    self.model.layers[i].self_attn.step = step
                    self.model.layers[i].self_attn.ori_sizes = ori_sizes
        elif self.training or "liuhaotian" in self.config._name_or_path:
            for i in range(len(self.model.layers)):
                self.model.layers[i].self_attn.image_cos = None
                self.model.layers[i].self_attn.image_sin = None
                self.model.layers[i].self_attn.hd_features = None
                self.model.layers[i].self_attn.image_range_list = None
                self.model.layers[i].self_attn.step = None
                self.model.layers[i].self_attn.ori_sizes = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        step: Optional[int] = 1.0,
        ori_sizes: Optional[List] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                hd_features,
                image_range_list
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            self.set_hd_info(hd_features, image_range_list, inputs_embeds, step=step, ori_sizes=ori_sizes)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        ori_sizes: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                hd_features,
                image_range_list
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            self.set_hd_info(hd_features, image_range_list, inputs_embeds, step=1.0, ori_sizes=ori_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
