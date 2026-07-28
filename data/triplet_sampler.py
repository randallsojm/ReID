import copy
import random
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
import os
import torchvision
import xml.etree.ElementTree as ET
import re

def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids 

import math
import xml.etree.ElementTree as ET
class CustomDataSet4CityFlow(Dataset):
    def __init__(self, xml_path, root_dir, is_train=True, transform=None,
                 id_list=None, split=None, query_fraction=0.3):
        self.root_dir = root_dir
        self.transform = transform
        records = self._parse_xml(xml_path)
        if id_list is not None:
            id_set = set(id_list)
            records = [r for r in records if r["vehicleID"] in id_set]
        if split in ("query", "gallery"):
            by_id = defaultdict(list)
            for r in records:
                by_id[r["vehicleID"]].append(r)
            selected = []
            for vid, recs in by_id.items():
                recs_sorted = sorted(recs, key=lambda r: r["imageName"])
                n_query = max(1, int(len(recs_sorted) * query_fraction))
                if split == "query":
                    selected.extend(recs_sorted[:n_query])
                else:
                    selected.extend(recs_sorted[n_query:])
            records = selected
        self.names = [r["imageName"] for r in records]
        raw_labels = [r["vehicleID"] for r in records]
        self.cams  = [r["cameraID"] for r in records]
        self.data_info = self.names
        if is_train:
            labels = sorted(set(raw_labels), key=lambda x: int(x))
            vid2pid = {vid: pid for pid, vid in enumerate(labels)}
            self.labels = [vid2pid[v] for v in raw_labels]
            print(f"[CityFlow] {len(self.names)} images, {len(labels)} identities (train)")
        else:
            self.labels = raw_labels
            print(f"[CityFlow] {len(self.names)} images ({split or 'eval'})")

    def _parse_xml(self, xml_path):
        try:
            with open(xml_path, encoding='gb2312') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(xml_path, encoding='utf-8') as f:
                content = f.read()

        if content.startswith('\ufeff'):
            content = content[1:]
        content = content.lstrip()
        if content.startswith('<?xml'):
            end = content.find('?>')
            if end != -1:
                content = content[end + 2:]
        content = content.lstrip()

        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            print("XML parse failed. First 120 chars, repr'd:", repr(content[:120]))
            raise

        items = root.find("Items")
        records = []
        for item in items.findall("Item"):
            records.append({
                "imageName": item.get("imageName"),
                "vehicleID": item.get("vehicleID"),
                "cameraID":  item.get("cameraID"),
            })
        return records

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.names[idx])
        image = torchvision.io.read_image(img_name)

        vid   = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].lstrip("c"))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor)) / 255.0)
        return img, vid, camid, 0  # no orientation info in CityFlow's XML -> static viewid
    
class CustomDataSet4VehicleX(Dataset):
    def __init__(self, xml_path, root_dir, is_train=True, transform=None,
                 pitch_min=None, pitch_max=None, n_view_bins=8,
                 id_list=None, split=None, query_fraction=0.3, n_cams=20):
        """
        id_list: set/list of vehicleID strings to restrict this dataset to
                 (e.g. train_ids or val_ids from split_vehiclex_ids.py).
        split: None (train mode, all images used) | 'query' | 'gallery'
               (for val identities, deterministically divides each identity's
               images so query/gallery don't overlap).
        query_fraction: fraction of each val identity's images assigned to query.
        n_cams: NEW. The checkpoint's camera-embedding table size (data['n_cams']
                at training time, e.g. 20 for a VeRi-776-trained checkpoint).
                VehicleX's own camera IDs are unrelated to VeRi's cameras and
                can exceed this range, causing an out-of-bounds embedding
                lookup at inference. camid is taken modulo n_cams in
                __getitem__ purely to stay in-bounds -- the resulting value
                carries no real camera-identity meaning for VehicleX and
                should not be relied on for same-camera junk removal (see the
                remove_junk fix in teste.py / the existing convention already
                established in main.py's eval_only branch).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_view_bins = n_view_bins
        self.n_cams = n_cams

        records = self._parse_xml(xml_path)
        if pitch_min is not None or pitch_max is not None:
            lo = pitch_min if pitch_min is not None else -999
            hi = pitch_max if pitch_max is not None else 999
            records = [r for r in records if lo <= r["pitch_deg"] <= hi]

        if id_list is not None:
            id_set = set(id_list)
            records = [r for r in records if r["vehicleID"] in id_set]

        if split in ("query", "gallery"):
            by_id = defaultdict(list)
            for r in records:
                by_id[r["vehicleID"]].append(r)
            selected = []
            for vid, recs in by_id.items():
                recs_sorted = sorted(recs, key=lambda r: r["imageName"])  # deterministic
                n_query = max(1, int(len(recs_sorted) * query_fraction))
                if split == "query":
                    selected.extend(recs_sorted[:n_query])
                else:
                    selected.extend(recs_sorted[n_query:])
            records = selected

        self.names  = [r["imageName"]   for r in records]
        raw_labels  = [r["vehicleID"]   for r in records]
        self.cams   = [r["cameraID"]    for r in records]
        self.orient = [r["orientation"] for r in records]
        self.colour = [r["colorID"]     for r in records]
        self.data_info = self.names 

        if is_train:
            labels = sorted(set(raw_labels), key=lambda x: int(x))
            vid2pid = {vid: pid for pid, vid in enumerate(labels)}
            self.labels = [vid2pid[v] for v in raw_labels]
            print(f"[VehicleX] {len(self.names)} images, {len(labels)} identities (train)")
        else:
            self.labels = raw_labels  # keep raw string IDs for query/gallery ID matching
            print(f"[VehicleX] {len(self.names)} images ({split or 'eval'})")

    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        items = root.find("Items")
        records = []
        for item in items.findall("Item"):
            camDis = float(item.get("camDis")); camHei = float(item.get("camHei"))
            records.append({
                "imageName":   item.get("imageName"),
                "vehicleID":   item.get("vehicleID"),
                "cameraID":    item.get("cameraID"),
                "colorID":     item.get("colorID"),
                "orientation": float(item.get("orientation")),
                "pitch_deg":   math.degrees(math.atan2(camHei, camDis)),
            })
        return records

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.names[idx])
        image = torchvision.io.read_image(img_name)

        vid    = np.int64(self.labels[idx])
        camid  = np.int64(int(self.cams[idx].lstrip("c")) % self.n_cams)  # FIX: was unclamped -> index-out-of-bounds crash on VehicleX's own camera IDs
        viewid = np.int64(int(self.orient[idx] // (360 / self.n_view_bins)) % self.n_view_bins)

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor)) / 255.0)
        return img, vid, camid, viewid

class CustomDataSet4VERIWILD(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = self.data_info.iloc[idx, 2]
        
        view_id = 0

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view :
            return img, vid, camid, view_id
        else:
            return img, vid, camid, 0


class CustomDataSet4VERIWILDv2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        self.data_info = pd.read_csv(csv_file, sep=' ', header=None)
        self.with_view = with_view
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.data_info.iloc[idx, 1]    

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data_info.iloc[idx, 0])
        image = torchvision.io.read_image(img_name)

        vid = self.data_info.iloc[idx, 1]
        camid = 0
        view_id = 0

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view:
            return img, vid, camid, view_id
        else:
            return img, vid, camid


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index in range(len(self.data_source.data_info)):
            pid = self.data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

class CustomDataSet4Veri776_withviewpont(Dataset):
    def __init__(self, image_list, root_dir, viewpoint_train, viewpoint_test, is_train=True, transform=None):
        self.viewpoint_train = pd.read_csv(viewpoint_train, sep=' ', header = None)
        self.viewpoint_test = pd.read_csv(viewpoint_test, sep=' ', header = None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        self.view = []
        conta_missing_images = 0
        if is_train == True:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line]
                if self.viewpoint_train[self.viewpoint_train.iloc[:, 0] == line].shape[0] ==0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1]) 
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            for line in lines:
                line = line.strip()
                view = self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line]
                if self.viewpoint_test[self.viewpoint_test.iloc[:, 0] == line].shape[0] == 0:
                    conta_missing_images += 1
                    continue
                view = int(view.iloc[0, -1])
                self.view.append(view)
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform
        print('Missed viewpoint for ', conta_missing_images, ' images!')
    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx])
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = np.int64(self.cams[idx].replace('c', ""))-1
        viewid = np.int64(self.view[idx])


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, viewid