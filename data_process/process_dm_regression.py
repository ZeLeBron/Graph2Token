# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
import re
from data_process.smiles2graph_regression import smiles2graph
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


class TrainCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])

    def __call__(self, batch):
        graph, instruction, text, text_values = zip(*batch)
        graphs = self.collater(graph)

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)

        text_values = torch.tensor(text_values).to(torch.float32)

        self.tokenizer.padding_side = 'right'
        text_tokens = self.tokenizer(text=text,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graphs, instruction_tokens, text_tokens, text_values


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])

    def __call__(self, batch):
        graph, instruction, text, text_values = zip(*batch)
        graphs = self.collater(graph)
        # deal with prompt
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(instruction,
                                            return_tensors='pt',
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            truncation=True,
                                            return_attention_mask=True)

        text_values = torch.tensor(text_values).to(torch.float32)
        return graphs, instruction_tokens, text, text_values


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class CheBIDataset(Dataset):
    def __init__(self, path, text_max_len, prompt=None):
        self.path = path
        self.text_max_len = text_max_len
        self.prompt = prompt

        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines][1:]

        self.smiles_list = []
        self.instruction_list = []
        self.text_list = []
        for line in lines:
            instruction, smiles, text = line.split('\t')  # qm9
            self.smiles_list.append(smiles)
            self.instruction_list.append(instruction)
            self.text_list.append(text)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        instruction = self.instruction_list[index]
        text_values = float(self.text_list[index])
        text = self.text_list[index]
        graph = smiles2data(smiles)
        return graph, instruction, text, text_values


class ProcessCheBIDM(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        if self.mode == 'pretrain':
            self.train_dataset = CheBIDataset(root + f'/train_sample_3600K.txt', text_max_len, self.prompt)
            self.val_dataset = CheBIDataset(root + '/valid.txt', text_max_len, self.prompt)
            self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len, self.prompt)
        else:
            self.train_dataset = CheBIDataset(root + f'/train.txt', text_max_len, self.prompt)
            self.val_dataset = CheBIDataset(root + '/valid.txt', text_max_len, self.prompt)
            self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer

    def train_dataloader(self):
        # assert self.mode == 'pretrain'
        # assert self.mode == 'ft'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )

        return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=6)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--inference_batch_size', type=int, default=16)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/finetune/qm9/homo')
        # parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        parser.add_argument('--graph_only', action='store_true', default=False)
        return parent_parser


