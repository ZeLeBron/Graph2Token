import os
import re
from typing import Any, Dict

import math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error

from model.QA_llama import Blip2Llama
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
import torch.distributed as dist
from model.help_funcs import AttrDict
from transformers import Adafactor


def cosine_lr_schedule(optimizer, step, max_step, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * step / max_step)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}

    # try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class QA_Trainer(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.llm_tune = args.llm_tune
        self.blip2opt = Blip2Llama(args.gin_num_layers,
                                   args.gin_hidden_dim,
                                   args.drop_ratio,
                                   args.tune_gnn,
                                   args.llm_tune,
                                   args.peft_dir,
                                   args.opt_model,
                                   args.prompt,
                                   args
                                   )
        self.save_hyperparameters(args)
        self.test_step_outputs = []

    def configure_optimizers(self):
        if self.args.optimizer == 'adafactor':
            print('Using adafactor optimizer')
            optimizer = Adafactor(
                self.parameters(),
                lr=1e-3,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False
            )
            self.scheduler = None
        else:
            self.trainer.fit_loop.setup_data()
            warmup_steps = self.args.warmup_steps
            optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
            if self.args.scheduler == 'linear_warmup_cosine_lr':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr,
                                                               self.args.init_lr, warmup_steps, self.args.warmup_lr)
            elif self.args.scheduler == 'linear_warmup_step_lr':
                self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr,
                                                             self.args.init_lr, self.args.lr_decay_rate,
                                                             self.args.warmup_lr, warmup_steps)
            elif self.args.scheduler == 'None':
                self.scheduler = None
            else:
                raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    @property
    def max_epochs(self):
        return self.fit_loop.max_epochs

    def save_predictions(self, predictions, targets, rmse):
        # assert len(predictions) == len(targets)
        # assert predictions.shape[0] == targets.shape[0]
        file_name = f"rmse_{rmse}.txt"
        with open(os.path.join(self.logger.log_dir, file_name), 'w', encoding='utf8') as f:
            # pass
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        # batch_size = batch[-1].input_ids.size(0)
        batch_size = batch[-1].size(0)
        # ============== Overall Loss ===================#
        loss = self.blip2opt(batch)
        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch):
        batch_size = batch[-1].size(0)
        loss = self.blip2opt(batch)
        # ============== Overall Loss =================== #
        self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def test_step(self, batch):
        graphs, instruction_tokens, texts, text_value = batch
        # ============== Captioning Results =================== #
        samples = {'graphs': graphs, 'instruction_tokens': instruction_tokens}
        predictions_text, predictions_values = self.blip2opt.generate(
            samples,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        self.test_step_outputs.append((predictions_text, predictions_values, texts, text_value))
        return predictions_text, predictions_values, texts, text_value

    def on_validation_epoch_start(self) -> None:
        self.list_predictions_text = []
        self.list_predictions_values = []
        self.list_targets_text = []
        self.list_targets_values = []

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.caption_eval_epoch != 0:
            return

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        list_predictions_text, list_predictions_values, list_targets_text, list_targets_values = zip(*outputs)

        predictions_text = [i for ii in list_predictions_text for i in ii]
        targets_text = [i for ii in list_targets_text for i in ii]

        predictions_values = torch.cat(list_predictions_values, dim=0).cpu().numpy()
        targets_values = torch.cat(list_targets_values, dim=0).cpu().numpy()

        all_predictions_text = [None for _ in range(self.trainer.world_size)]
        all_targets_text = [None for _ in range(self.trainer.world_size)]
        all_predictions_values = [None for _ in range(self.trainer.world_size)]
        all_targets_values = [None for _ in range(self.trainer.world_size)]
        try:
            dist.all_gather_object(all_predictions_text, predictions_text)
            dist.all_gather_object(all_targets_text, targets_text)
            dist.all_gather_object(all_predictions_values, predictions_values)
            dist.all_gather_object(all_targets_values, targets_values)
        except RuntimeError:
            all_predictions_text = [predictions_text]
            all_targets_text = [targets_text]
            all_predictions_values = [predictions_values]
            all_targets_values = [targets_values]

        if self.global_rank == 0:
            all_predictions_text = [i for ii in all_predictions_text for i in ii]
            all_targets_text = [i for ii in all_targets_text for i in ii]
            all_predictions_values = [i for ii in all_predictions_values for i in ii]
            all_targets_values = [i for ii in all_targets_values for i in ii]

            mse = mean_squared_error(np.array(all_predictions_values), np.array(all_targets_values))
            rmse = np.sqrt(mse)

            self.log("rmse", rmse, sync_dist=False)
            self.save_predictions(all_predictions_text, all_targets_text, rmse)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', type=bool, default=True)
        # OPT
        parser.add_argument('--opt_model', type=str, default="../LLM/vicuna-7b-v1.5")
        parser.add_argument('--prompt', type=str, default=None)
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=5)
        parser.add_argument('--min_len', type=int, default=2)
        parser.add_argument('--llm_tune', action='store_true', default=False)
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=100)
        # quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        # lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr',
                            help='type of scheduler')  # or linear_warmup_step_lr
        parser.add_argument('--optimizer', type=str, default='adamw', help='type of scheduler')
        parser.add_argument('--finetune_path', type=str,
                            default='../LLM/vicuna-7b-v1.5')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=1)
        return parent_parser
