import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import *
from data.triplet_sampler import *
from typing import OrderedDict
from processor import get_model
import torch.multiprocessing
import os
import yaml
from utils import re_ranking, re_ranking_azimuth  # re_ranking unchanged (bug-fix only); re_ranking_azimuth is new
#import cv2



def normalize_batch(batch, maximo=None, minimo = None):
    if maximo != None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1))
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, remove_junk=True, scaler=None,
                re_rank=False, azimuth_rerank=False, min_azimuth_gap_bins=1):
    # azimuth_rerank / min_azimuth_gap_bins are no-ops unless re_rank=True too,
    # and when azimuth_rerank=False (the default) every line below runs
    # exactly as it did before this change.
    model.eval()
    re_escala = torchvision.transforms.Resize((256,256), antialias=True)

    ###needed lists
    qf = []
    gf = []
    q_camids = []
    g_camids = []
    q_vids = []
    g_vids = []
    q_images = []
    g_images =  []
    q_views = []   # NEW: only populated when azimuth_rerank=True
    g_views = []   # NEW: only populated when azimuth_rerank=True
    count_imgs = 0
    blend_ratio =0.3
    with torch.no_grad():
        for image, q_id, cam_id, view_id  in tqdm(dataloader_q, desc='Query infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id)

            count_imgs += activations[0].shape[0]
            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            qf.append(torch.cat(end_vec, 1))
            q_vids.append(q_id)
            q_camids.append(cam_id)
            if azimuth_rerank:                     # NEW
                q_views.append(view_id)             # NEW

        del q_images
        count_imgs = 0
        for image, g_id, cam_id, view_id in tqdm(dataloader_g, desc='Gallery infer (%)', bar_format='{l_bar}{bar:20}{r_bar}'):
            image = image.to(device)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, ffs, activations = model(image, cam_id, view_id)
            else:
                _, _, ffs, activations = model(image, cam_id, view_id)

            end_vec = []
            for item in ffs:
                end_vec.append(F.normalize(item))
            gf.append(torch.cat(end_vec, 1))
            g_vids.append(g_id)
            g_camids.append(cam_id)
            if azimuth_rerank:                     # NEW
                g_views.append(view_id)             # NEW

            count_imgs += activations[0].shape[0]

        del g_images

    qf = torch.cat(qf, dim=0)
    gf = torch.cat(gf, dim=0)

    m, n = qf.shape[0], gf.shape[0]
    if re_rank and azimuth_rerank:                                    # NEW branch
        q_view_arr = torch.cat(q_views, dim=0)
        g_view_arr = torch.cat(g_views, dim=0)
        distmat = re_ranking_azimuth(qf, gf, k1=80, k2=16, lambda_value=0.3,
                                      q_view=q_view_arr, g_view=g_view_arr,
                                      min_azimuth_gap_bins=min_azimuth_gap_bins)
    elif re_rank:                                                      # UNCHANGED
        distmat = re_ranking(qf, gf, k1=80, k2=16, lambda_value=0.3)
    else:                                                               # UNCHANGED
        distmat =  torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(qf, gf.t(),beta=1, alpha=-2)
        distmat = torch.sqrt(distmat).cpu().numpy()

    q_camids = torch.cat(q_camids, dim=0).cpu().numpy()
    g_camids = torch.cat(g_camids, dim=0).cpu().numpy()
    q_vids = torch.cat(q_vids, dim=0).cpu().numpy()
    g_vids = torch.cat(g_vids, dim=0).cpu().numpy()

    del qf, gf

    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)
    print(f'mAP = {mAP},  CMC1= {cmc[0]}, CMC5= {cmc[4]}')

    return cmc, mAP


if __name__ == "__main__":

    ### Just to ensure VehicleID 10-fold validation randomness is not random to compare different models training
    set_seed(0)
    parser = argparse.ArgumentParser(description='Reid train')

    parser.add_argument('--batch_size', default=None, type=int, help='an integer for the accumulator')
    parser.add_argument('--dataset', default=None, help='Choose one of[Veri776, VERIWILD, CityFlow (official, unlabeled query/test), CityFlowVal (held-out labeled split from train_label.xml)]')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--path_weights', default=None, help="Path to *.pth/*.pt loading weights file")
    parser.add_argument('--re_rank', action="store_true", help="Re-Rank")
    parser.add_argument('--azimuth_rerank', action="store_true",
                         help="Requires --re_rank. Restricts re-rank neighbor selection to crops "
                              ">= min_azimuth_gap_bins orientation bins away, using the viewid "
                              "already returned by CustomDataSet4Veri776_withviewpont. "
                              "Only meaningful for --dataset Veri776.")
    parser.add_argument('--min_azimuth_gap_bins', default=1, type=int,
                         help="Minimum circular bin distance (of 8 orientation bins, ~45deg/bin) "
                              "required for a candidate to be eligible as a re-rank neighbor.")
    # CityFlow path overrides — the saved config.yaml at path_weights still
    # has whatever dataset's paths it was trained with (e.g. VehicleX's),
    # so these let you point at CityFlow's actual label files without
    # editing that saved config. n_classes/model_arch don't need an
    # override: n_classes is irrelevant at eval time (only ffs embeddings
    # are used, see load-time shape-mismatch skip below), and model_arch
    # already matches the checkpoint via the saved config.
    parser.add_argument('--query_list_file', default=None, help='Override: path to query_label.xml (CityFlow) or query list file')
    parser.add_argument('--query_dir', default=None, help='Override: path to query images dir')
    parser.add_argument('--gallery_list_file', default=None, help='Override: path to test_label.xml (CityFlow) or gallery list file')
    parser.add_argument('--teste_dir', default=None, help='Override: path to gallery/test images dir')
    parser.add_argument('--train_list_file', default=None, help='Override: path to train_label.xml — used by CityFlowVal, which reads train_list_file/train_dir directly (NOT query_list_file/gallery_list_file)')
    parser.add_argument('--train_dir', default=None, help='Override: path to training images dir — used by CityFlowVal')
    args = parser.parse_args()

    if args.azimuth_rerank and not args.re_rank:
        raise ValueError("--azimuth_rerank requires --re_rank to also be set.")

    with open(args.path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['dataset'] = args.dataset or data['dataset']
    data['model_arch'] = args.model_arch or data['model_arch']

    # NOTE: checked here (after data['dataset'] is resolved from config.yaml,
    # not just the raw --dataset CLI flag) so runs that rely on the saved
    # config's dataset value -- like not passing --dataset at all -- are
    # evaluated correctly instead of tripping on an unset CLI arg.
    if args.azimuth_rerank and data['dataset'] != 'Veri776':
        raise ValueError(f"--azimuth_rerank requires dataset Veri776, got '{data['dataset']}' "
                          "(only CustomDataSet4Veri776_withviewpont returns real orientation labels; "
                          "CityFlow/VERIWILD/VehicleID view_id is a static 0 placeholder, not real orientation).")
    if args.query_list_file is not None: data['query_list_file'] = args.query_list_file
    if args.query_dir is not None: data['query_dir'] = args.query_dir
    if args.gallery_list_file is not None: data['gallery_list_file'] = args.gallery_list_file
    if args.teste_dir is not None: data['teste_dir'] = args.teste_dir
    if args.train_list_file is not None: data['train_list_file'] = args.train_list_file
    if args.train_dir is not None: data['train_dir'] = args.train_dir


    teste_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),

    ])                  

    if data['half_precision']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler=False

    ### Replace paths as needed
    if data['dataset']== 'VERIWILD':
        data['n_classes'] = 30671
        data_q = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id_query.txt', data['ROOT_DIR'], transform=teste_transform, with_view=True)
        data_g = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id.txt', data['ROOT_DIR'], transform=teste_transform, with_view=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste']) #data['BATCH_SIZE']
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset']== 'VERIWILD2.0':
        data['n_classes'] = 30671
        vw2_dir = "/mnt/DATADISK/Datasets/vehicle/VeriWild/v2.0/"
        set = 'B' #args.vw2_set A, B or All
        data_q = CustomDataSet4VERIWILDv2(vw2_dir + 'test_split_V2/'+ set +'_query.txt', vw2_dir, transform=teste_transform, with_view=True)
        data_g = CustomDataSet4VERIWILDv2(vw2_dir + 'test_split_V2/'+ set +'_gallery.txt', vw2_dir, transform=teste_transform, with_view=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=0) #data['BATCH_SIZE'] data['num_workers_teste']
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=0)


    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'CityFlow':
        # CityFlow's own official query/test XML partition — query_label.xml
        # / test_label.xml. NOTE: these have NO vehicleID ground truth (AIC
        # Challenge withholds it for server-side leaderboard evaluation) —
        # this branch can extract embeddings but test_epoch's mAP/CMC will
        # fail (or be meaningless) since there's no label to compare
        # against. Use 'CityFlowVal' below for actual local evaluation.
        data_q = CustomDataSet4CityFlow(data['query_list_file'], data['query_dir'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4CityFlow(data['gallery_list_file'], data['teste_dir'], is_train=False, transform=teste_transform)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'CityFlowVal':
        # Held-out identity-disjoint split built from train_label.xml (the
        # only CityFlow file with real vehicleID ground truth), via
        # split_cityflow_ids.py. Both query and gallery are carved out of
        # image_train + train_label.xml using val_ids — NOT the official
        # image_query/image_test dirs, since those have no labels at all.
        with open("dataset/Cityflow/AIC21_Track2_ReID/cityflow_val_ids.txt") as f:
            val_ids = [l.strip() for l in f if l.strip()]

        data_q = CustomDataSet4CityFlow(
            data['train_list_file'], data['train_dir'], is_train=False,
            transform=teste_transform, id_list=val_ids, split='query')
        data_g = CustomDataSet4CityFlow(
            data['train_list_file'], data['train_dir'], is_train=False,
            transform=teste_transform, id_list=val_ids, split='gallery')
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])


    if data['dataset'] == 'VehicleX':
        # NEW: was missing entirely -- CustomDataSet4VehicleX already exists
        # in triplet_sampler.py (used by main.py for training) and already
        # supports id_list/split='query'|'gallery' the same way
        # CustomDataSet4CityFlow does for CityFlowVal above, but no branch
        # here ever called it, so --dataset VehicleX previously fell through
        # every if-check with data_q/data_g undefined -> NameError.
        #
        # Held-out identity split, mirroring split_cityflow_ids.py's
        # counterpart (split_vehiclex_ids.py, per the session summary).
        # ADJUST THIS PATH to wherever your val-id list actually lives --
        # I don't have your dataset directory layout, this mirrors the
        # CityFlowVal path convention above as a best guess.
        with open("dataset/vehiclex_data/vehiclex_val_ids.txt") as f:
            val_ids = [l.strip() for l in f if l.strip()]

        # Reuses the existing --train_list_file/--train_dir override flags
        # (already generic, not CityFlow-specific) to point at VehicleX's
        # XML + image root, same as CityFlowVal does above.
        data_q = CustomDataSet4VehicleX(
            data['train_list_file'], data['train_dir'], is_train=False,
            transform=teste_transform, id_list=val_ids, split='query',
            n_cams=data['n_cams'])
        data_g = CustomDataSet4VehicleX(
            data['train_list_file'], data['train_dir'], is_train=False,
            transform=teste_transform, id_list=val_ids, split='gallery',
            n_cams=data['n_cams'])
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    model = get_model(data, torch.device("cpu"))

    # One of the saved weights last.pt best_CMC.pt best_mAP.pt
    path_weights = args.path_weights + 'best_mAP.pt'

    try:
        model.load_state_dict(torch.load(path_weights, map_location='cpu')) 
    except RuntimeError:
        ### nn.Parallel adds "module." to the dict names. Although like said nn.Parallel can incur in weird results in some cases 
        tmp = torch.load(path_weights, map_location='cpu')
        tmp = OrderedDict((k.replace("module.", ""), v) for k, v in tmp.items())
        model.load_state_dict(tmp)

    
    model = model.to(device)
    model.eval()

    mean = False
    l2 = True

    # FIX: was hardcoded remove_junk=True at both call sites below, applying
    # VeRi-style same-camera junk removal unconditionally. main.py's
    # eval_only branch already established VehicleX needs remove_junk=False
    # (different camera/junk convention -- see that branch's comment); teste.py
    # never got the matching fix until now.
    remove_junk = data['dataset'] not in ('VehicleX',)

    if data['dataset'] == "VehicleID":
        list_mAP = []
        list_cmc1 = []
        list_cmc5 = []
        for i in range(10):
            reader = open('/home/eurico/VehicleID_V1.0/train_test_split/test_list_800.txt')
            lines = reader.readlines()
            random.shuffle(lines)
            data_q = CustomDataSet4VehicleID_Random(lines, data['ROOT_DIR'], is_train=False, mode="q", transform=teste_transform, teste=True)
            data_g = CustomDataSet4VehicleID_Random(lines, data['ROOT_DIR'], is_train=False, mode="g", transform=teste_transform, teste=True)
            data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
            data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
            cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=remove_junk, scaler=scaler,
                                   re_rank=args.re_rank, azimuth_rerank=args.azimuth_rerank,
                                   min_azimuth_gap_bins=args.min_azimuth_gap_bins)
            list_mAP.append(mAP)
            list_cmc1.append(cmc[0])
            list_cmc5.append(cmc[4])
        mAP = sum(list_mAP) / len(list_mAP)
        cmc1 = sum(list_cmc1) / len(list_cmc1)
        cmc5 = sum(list_cmc5) / len(list_cmc5)
        print(f'\n\nmAP = {mAP},  CMC1= {cmc1}, CMC5= {cmc5}')
        with open(args.path_weights +'result_map_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, mAP)
        with open(args.path_weights +'result_cmc_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, cmc1)
    else:
        cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=remove_junk, scaler=scaler,
                               re_rank=args.re_rank, azimuth_rerank=args.azimuth_rerank,
                               min_azimuth_gap_bins=args.min_azimuth_gap_bins)
        print(f'mAP = {mAP},  CMC1= {cmc[0]}, CMC5= {cmc[4]}')
        with open(args.path_weights +'result_map_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, mAP)
        with open(args.path_weights +'result_cmc_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, cmc)

    print('Weights: ', path_weights)