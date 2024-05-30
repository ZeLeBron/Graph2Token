"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from torch.nn import MSELoss
from torch_geometric.utils import to_dense_batch

from model.blip2 import Blip2Base
from transformers import LlamaTokenizer, BitsAndBytesConfig, AutoTokenizer
from model.modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification

llama_model_list = [
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-7b-hf",
]

local_rank = int(os.environ.get('LOCAL_RANK', '0'))
device_map = {'': local_rank}


def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Llama(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    def __init__(
            self,
            gin_num_layers,
            gin_hidden_dim,
            gin_drop_ratio,
            tune_gnn=False,
            lora_tuning=False,
            peft_dir='',
            llm_model="decapoda-research/llama-7b-hf",
            prompt="",
            args=None,
    ):
        super().__init__()
        self.gin_hidden_dim = gin_hidden_dim
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")

        # initialize opt model
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['<mol>']})
        self.llm_tokenizer.mol_token_id = self.llm_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        self.lora_tuning = lora_tuning
        if self.lora_tuning:
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_dir, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                         lora_dropout=0.1)
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            # for name, param in self.llm_model.named_parameters():
            #     if name == 'score.weight':
            #         param.requires_grad = True

        # fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer.eos_token_id
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        self.down_graph_token = nn.Linear(4096, 128)
        self.activate = nn.ReLU()
        self.score = self.score = nn.Linear(128, 1, bias=False)

        # fixme: no prompt yet
        self.prompt = prompt

    def forward(self, batch):
        graphs, instruction_tokens, text_tokens, text_values = batch
        h_graph = self.graph_encoder(graphs)
        device = h_graph.device

        graph_atts_mask = torch.ones(h_graph.size()[0], dtype=torch.long).to(device)
        graph_atts_mask = graph_atts_mask.unsqueeze(1)

        instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
        graph_inputs_llm = h_graph
        graph_inputs_llm = graph_inputs_llm.unsqueeze(1)
        inputs_embeds = torch.cat([instruction_embeds, graph_inputs_llm], dim=1)
        attention_mask = torch.cat([instruction_tokens.attention_mask, graph_atts_mask], dim=1)

        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(graph_atts_mask.size(), dtype=torch.long).to(device).fill_(-100)
        )
        instruct_targets = torch.ones(
            instruction_tokens.attention_mask.shape, dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat((instruct_targets, empty_targets, targets), dim=1)

        outputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_embeds, outputs_embeds], dim=1)
        attention_mask = torch.cat([attention_mask, text_tokens.attention_mask], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            # use_cache=False,
        )

        output = outputs.hidden_states
        graph_token_embedding = output[:, -1, :]
        down_graph_token_embedding = self.activate(self.down_graph_token(graph_token_embedding))
        logits = self.score(down_graph_token_embedding)
        loss_fct = MSELoss()
        loss_mse = loss_fct(logits.squeeze(), text_values.squeeze())
        loss = outputs.loss + loss_mse
        return {"loss": loss}

    @torch.no_grad()
    def generate(
            self,
            samples,
            do_sample=False,
            num_beams=5,
            max_length=128,
            min_length=1,
            max_new_tokens=128,
            min_new_tokens=32,
            repetition_penalty=1.2,
            length_penalty=1.0,
            num_captions=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        instruction_tokens = samples['instruction_tokens']
        with self.maybe_autocast():
            h_graph = self.graph_encoder(graphs)

            graph_atts_mask = torch.ones(h_graph.size()[0], dtype=torch.long).to(h_graph.device)
            graph_atts_mask = graph_atts_mask.unsqueeze(1)

            instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)
            graph_inputs_llm = h_graph
            graph_inputs_llm = graph_inputs_llm.unsqueeze(1)
            inputs_embeds = torch.cat([instruction_embeds, graph_inputs_llm], dim=1)
            attention_mask = torch.cat([instruction_tokens.attention_mask, graph_atts_mask], dim=1)

            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                # use_cache=False,
            )

            output = outputs.hidden_states
            graph_token_embedding = output[:, -1, :]
            down_graph_token_embedding = self.activate(self.down_graph_token(graph_token_embedding))

            logits = self.score(down_graph_token_embedding)

            outputs_text_token = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=do_sample,
                num_beams=num_beams,
                max_length=max_length,
                # min_length=min_length,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.llm_tokenizer.batch_decode(outputs_text_token, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text, logits
