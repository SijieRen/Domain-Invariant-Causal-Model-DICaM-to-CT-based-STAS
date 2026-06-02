# coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import random
import xlrd
from torchvision import transforms
import pandas as pd


def _read_sheet_dataframe(excel_path, sheet_name):
    return pd.read_excel(excel_path, sheet_name=sheet_name)


def _resolve_data_root(args):
    # 1) explicit CLI arg
    data_root = getattr(args, 'data_root', None) if args else None
    if data_root:
        return os.path.abspath(data_root)

    # 2) project default
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_root = os.path.abspath(os.path.join(project_root, "..", "dataset"))
    if os.path.isdir(default_root):
        return default_root

    # 3) local sibling dataset folder used in this workspace
    sibling_root = os.path.abspath(os.path.join(project_root, "..", "0111_toZicheng"))
    if os.path.isdir(sibling_root):
        return sibling_root

    # 4) final fallback (keeps previous behavior)
    return default_root


def _normalize_rel_path(rel_path):
    rel_norm = str(rel_path).replace("\\", "/").lstrip("/")
    known_prefixes = (
        "../dataset/",
        "./dataset/",
        "dataset/",
        "../0111_toZicheng/",
        "./0111_toZicheng/",
        "0111_toZicheng/",
    )
    for prefix in known_prefixes:
        if rel_norm.startswith(prefix):
            rel_norm = rel_norm[len(prefix):]
            break
    return rel_norm


def _apply_dataset_aliases(rel_norm):
    # Some local exports use Nomask folder names.
    aliases = [rel_norm]
    aliases.append(rel_norm.replace("0727_zhongshan_2D_maskOnly_bg0", "0727_zhongshan_2D_Nomask"))
    aliases.append(rel_norm.replace("0727_zhongliu_2D_maskOnly_bg0", "0727_zhongliu_2D_Nomask"))
    aliases.append(rel_norm.replace("0727_jiashan_2D_maskOnly_bg0", "0727_jiashan_2D_Nomask"))
    # de-duplicate while preserving order
    return list(dict.fromkeys(aliases))

class Dataloader_2D(Dataset):
    def __init__(self, data_index=0, fold='train', transform="None", args=None, dataframe=None):
        self.data_index = data_index
        self.fold = fold
        self.transform = transform
        self.args = args
        if dataframe is not None:
            worksheet = dataframe
        else:
            worksheet = _read_sheet_dataframe(self.data_index, self.fold)
        self.id = []
        self.samples = []
        self.labels = []
        self.A_1 = []
        self.A_2 = []
        self.Machine = []
        self.Hospital = []
        data_root = _resolve_data_root(self.args)
        # Excel 中可能含旧机器的绝对路径，需替换为 data_root 下的 dataset 路径
        _OLD_ABSOLUTE_PREFIXES = ("/home/u20111510066/sijie_data/", "/home/u20111510066/sijie_data")
        for _, row in worksheet.iterrows():
            rel_path = row.iloc[0]
            rel_norm = _normalize_rel_path(rel_path)
            if os.path.isabs(rel_path):
                for prefix in _OLD_ABSOLUTE_PREFIXES:
                    if rel_path.startswith(prefix):
                        rel_path = rel_path[len(prefix):].lstrip("/")
                        rel_norm = _normalize_rel_path(rel_path)
                        full_path = os.path.join(data_root, rel_norm)
                        break
                else:
                    full_path = rel_path  # 未匹配前缀，保持原绝对路径
            else:
                rel_norm = _normalize_rel_path(rel_path)
                full_path = os.path.join(data_root, rel_norm)

            if not os.path.exists(full_path):
                candidate_paths = [os.path.join(data_root, p) for p in _apply_dataset_aliases(rel_norm)]
                for candidate_path in candidate_paths:
                    if os.path.exists(candidate_path):
                        full_path = candidate_path
                        break
            self.samples.append(os.path.normpath(full_path))
            self.Machine.append(int(row.iloc[2]))#
            self.Hospital.append(int(row.iloc[3]))#
            self.labels.append(int(row.iloc[4]))# 0/1
            self.A_1.append(row.iloc[9:11].tolist())#load gender and age
            if self.args.if_randomTest==2:
                self.A_2.append(row.iloc[11:28].tolist())#load the bi-predict in A2
            else:
                self.A_2.append(row.iloc[11:25].tolist())#load the bi-predict in A2
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = Image.open(samples).convert('L')
        if self.transform is not None:
            samples = self.transform(samples)
        labels = self.labels[idx]
        A_1 = self.A_1[idx]
        A_2 = self.A_2[idx]
        Machine = self.Machine[idx]
        Hospital = self.Hospital[idx]


        return samples,\
               torch.from_numpy(np.array(labels).astype("int")),\
               torch.from_numpy(np.array(A_1).astype("float32")),\
               torch.from_numpy(np.array(A_2).astype("float32")),\
               torch.from_numpy(np.array(Machine).astype("int")),\
               torch.from_numpy(np.array(Hospital).astype("int"))

    def __len__(self):
        return len(self.samples)
    

def read_nii_file(nii_file_path):
    sitk_data = sitk.ReadImage(nii_file_path)
    return sitk.GetArrayFromImage(sitk_data), sitk_data 

class Dataloader_3D(Dataset):
    def __init__(self, data_index=0, data_root="", fold='train', transform="None"):
        self.data_index = data_index
        self.root = data_root
        self.mode = fold
        self.transform = transform
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        if self.mode == 'train':
            worksheet = workbook.sheet_by_name('train')
        else:
            worksheet = workbook.sheet_by_name('test')
        self.id = []
        self.samples = []
        self.labels = []
        self.A_1 = []
        self.A_2 = []
        self.Machine = []
        self.Hospital = []
        for i in range(2, worksheet.nrows):
            self.samples.append(os.path.join("..", worksheet.row_values(i)[0]))#pat of the samples
            self.Machine.append(int(worksheet.row_values(i)[2]))#
            self.Hospital.append(int(worksheet.row_values(i)[3]))#
            self.labels.append(int(worksheet.row_values(i)[8]))# 0/1
            self.A_1.append(int(worksheet.row_values(i)[12:14]))#load gender and age
            self.A_2.append(int(worksheet.row_values(i)[16:25]))#load the bi-predict in A2
    
    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        samples = self.samples[idx]
        samples = Image.open(samples).convert('L')
        if self.transform is not None:
            samples = self.transform(samples)
        labels = self.labels[idx]
        A_1 = self.A_1[idx]
        A_2 = self.A_2[idx]
        Machine = self.Machine[idx]
        Hospital = self.Hospital[idx]


        return samples,\
               torch.from_numpy(np.array(labels).astype("int")),\
               torch.from_numpy(np.array(A_1).astype("int")),\
               torch.from_numpy(np.array(A_2).astype("int")),\
               torch.from_numpy(np.array(Machine).astype("int")),\
               torch.from_numpy(np.array(Hospital).astype("int"))

    def __len__(self):
        return len(self.samples)
