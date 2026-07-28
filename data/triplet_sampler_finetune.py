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


def train_collate_fn(batch):
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)

    return torch.stack(imgs, dim=0), pids, camids, viewids 

        
class CustomDataSet4VERIWILD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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
        
        view_id = 0 #self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view :
            return img, vid, camid, view_id
        else:
            return img, vid, camid, 0




import math
import xml.etree.ElementTree as ET

class CustomDataSet4VehicleX(Dataset):
    def __init__(self, xml_path, root_dir, is_train=True, transform=None,
                 pitch_min=None, pitch_max=None, n_view_bins=8,
                 id_list=None, split=None, query_fraction=0.3):
        """
        id_list: set/list of vehicleID strings to restrict this dataset to
                 (e.g. train_ids or val_ids from split_vehiclex_ids.py).
        split: None (train mode, all images used) | 'query' | 'gallery'
               (for val identities, deterministically divides each identity's
               images so query/gallery don't overlap).
        query_fraction: fraction of each val identity's images assigned to query.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.n_view_bins = n_view_bins

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
        camid  = np.int64(self.cams[idx].lstrip("c"))
        viewid = np.int64(int(self.orient[idx] // (360 / self.n_view_bins)) % self.n_view_bins)

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor)) / 255.0)
        return img, vid, camid, viewid
    
class OrientationAwareIdentitySampler(Sampler):
    """
    Biases WITHIN-identity instance selection toward the empirically hardest
    azimuth gap (~144°, from car1_dataset --angle_gap_map results), so
    triplet loss sees more hard-positive pairs at that gap during training.
    """
    def __init__(self, data_source, batch_size, num_instances,
                 target_gap=144, hard_ratio=0.5):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.target_gap = target_gap
        self.hard_ratio = hard_ratio

        self.index_dic = defaultdict(list)
        for index in range(len(data_source.names)):
            pid = data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def _gap(self, a, b):
        d = abs(a - b)
        return min(d, 360 - d)

    def _select_instances(self, idxs):
        if len(idxs) <= self.num_instances:
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, size=self.num_instances, replace=True))
            random.shuffle(idxs)
            return idxs

        orients = {i: self.data_source.orient[i] for i in idxs}
        n_hard = int(self.num_instances * self.hard_ratio)

        pool = list(idxs)
        selected = [random.choice(pool)]
        pool.remove(selected[0])

        while len(selected) < n_hard and pool:
            def worst_case_score(i):
                # deviation from target_gap against EVERY already-selected instance,
                # not just the average — we want every pair to be close to target,
                # not just the mean of all pairs
                deviations = [abs(self._gap(orients[i], orients[s]) - self.target_gap)
                              for s in selected]
                return -max(deviations)   # minimize the WORST pairwise deviation

            best = max(pool, key=worst_case_score)
            selected.append(best)
            pool.remove(best)

        remaining = self.num_instances - len(selected)
        if remaining > 0 and pool:
            fill = random.sample(pool, min(remaining, len(pool)))
            selected.extend(fill)
        while len(selected) < self.num_instances:
            selected.append(random.choice(idxs))

        random.shuffle(selected)
        return selected

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            batch = self._select_instances(idxs)
            batch_idxs_dict[pid].append(batch)

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
    
class ColourAwareIdentitySampler(Sampler):
    """
    Biases WHICH identities co-occur in a batch toward same-colour groups,
    so triplet loss sees hard negatives (different vehicle, same colour)
    during training, matching the 10<->5 / 12<->8 style FP confusions found.
    """
    def __init__(self, data_source, batch_size, num_instances, hard_colour_ratio=0.4):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.hard_colour_ratio = hard_colour_ratio

        self.index_dic = defaultdict(list)
        for index in range(len(data_source.names)):
            pid = data_source.get_class(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.pid_colour = {}
        for pid in self.pids:
            first_idx = self.index_dic[pid][0]
            self.pid_colour[pid] = data_source.colour[first_idx]

        self.colour_groups = defaultdict(list)
        for pid, colour in self.pid_colour.items():
            self.colour_groups[colour].append(pid)

        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def _select_pids_for_batch(self, avai_pids):
        n_hard = int(self.num_pids_per_batch * self.hard_colour_ratio)

        avai_set = set(avai_pids)
        selected = []

        seed = random.choice(avai_pids)
        selected.append(seed)
        avai_set.discard(seed)

        seed_colour = self.pid_colour[seed]
        same_colour_candidates = [p for p in self.colour_groups[seed_colour] if p in avai_set]
        random.shuffle(same_colour_candidates)

        while len(selected) < n_hard and same_colour_candidates:
            selected.append(same_colour_candidates.pop())
            avai_set.discard(selected[-1])

        remaining_pool = list(avai_set)
        random.shuffle(remaining_pool)
        remaining_needed = self.num_pids_per_batch - len(selected)
        selected.extend(remaining_pool[:remaining_needed])

        return selected

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = list(np.random.choice(idxs, size=self.num_instances, replace=True))
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
            selected_pids = self._select_pids_for_batch(avai_pids)
            for pid in selected_pids:
                if not batch_idxs_dict[pid]:
                    if pid in avai_pids: avai_pids.remove(pid)
                    continue
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
    
class CustomDataSet4VERIWILDv2(Dataset):
    """VeriWild 2.0 dataset."""

    def __init__(self, csv_file, root_dir, transform=None, with_view=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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
        camid = 0 #self.data_info.iloc[idx, 2]
        view_id = 0 # = self.data_info.iloc[idx, 3]

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.with_view:
            return img, vid, camid, view_id
        else:
            return img, vid, camid



class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

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

        # estimate number of examples in an epoch
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

        
class CustomDataSet4Market1501(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
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
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

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
        vid = np.int64(self.labels[idx])  # works directly since raw IDs are numeric strings like "00001"
        camid = np.int64(self.cams[idx].split('s')[0].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid     

       
 


class CustomDataSet4Veri776(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.data_info = pd.read_xml(csv_file, sep=' ', header=None)
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        self.cams = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])     
            labels = sorted(set(self.labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
                # print(pid, id, 'debug')
        else:
            for line in lines:
                line = line.strip()
                self.names.append(line)
                self.labels.append(line.split('_')[0])
                self.cams.append(line.split('_')[1])      
        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

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
        camid = np.int64(self.cams[idx].replace('c', ""))


        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0 






class CustomDataSet4Veri776_withviewpont(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, viewpoint_train, viewpoint_test, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

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

class CustomDataSet4VehicleID_Random(Dataset):
    def __init__(self, lines, root_dir, is_train=True, mode=None, transform=None, teste=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_info = []
        self.names = []
        self.labels = []
        self.teste = teste
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        ### no camera information
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)
        if self.teste:
            return img, vid, camid, 0
        else:
            return img, vid, camid





class CustomDataSet4VehicleID(Dataset):
    def __init__(self, image_list, root_dir, is_train=True, mode=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        reader = open(image_list)
        lines = reader.readlines()
        self.data_info = []
        self.names = []
        self.labels = []
        if is_train == True:
            for line in lines:
                line = line.strip()
                name = line[:7] 
                vid = line[8:]
                self.names.append(name)
                self.labels.append(vid)   
            labels = sorted(set(self.labels))
            print("ncls: ",len(labels))
            for pid, id in enumerate(labels):
                idxs = [i for i, v in enumerate(self.labels) if v==id] 
                for j in idxs:
                    self.labels[j] = pid
        else:
            print("Dataload Test mode: ", mode)
            vid_container = set()
            for line in lines:
                line = line.strip()
                name = line[:7]
                vid = line[8:]
                # random.shuffle(dataset)
                if mode=='g':  
                    if vid not in vid_container:
                        vid_container.add(vid)
                        self.names.append(name)
                        self.labels.append(vid)
                else:
                    if vid not in vid_container:
                        vid_container.add(vid)
                    else:
                        self.names.append(name)
                        self.labels.append(vid)

        self.data_info = self.names        
        self.root_dir = root_dir
        self.transform = transform

    def get_class(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.names[idx]+ ".jpg")
        image = torchvision.io.read_image(img_name)
        vid = np.int64(self.labels[idx])
        camid = idx #np.int64(self.cams[idx].replace('c', ""))

        if self.transform:
            img = self.transform((image.type(torch.FloatTensor))/255.0)

        return img, vid, camid, 0


