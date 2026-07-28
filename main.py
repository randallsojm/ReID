import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from data.triplet_sampler import *
from loss.losses import triplet_loss_fastreid
from loss.losses import SupConLoss
from lr_scheduler.sche_optim import make_optimizer, make_warmup_scheduler
import argparse
import torch.multiprocessing
import yaml
import os
from tensorboard_log import Logger
from processor import get_model, train_epoch, test_epoch
import torchvision.transforms.v2 as T
import xml.etree.ElementTree as ET
import math
from collections import Counter
import sys


torch.multiprocessing.set_sharing_strategy('file_system')

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES']="-1"

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

def load_pretrained_backbone(model, checkpoint_path, device):
    """
    Warm-starts a freshly-constructed model from a checkpoint, copying only
    tensors whose key AND shape match (e.g. a VeRi-pretrained checkpoint
    loaded into a model with a different n_classes — the classifier head(s)
    will be shape-mismatched and skipped, left randomly initialized).
    Opt-in only: called from main() below iff data['finetune_from'] is set.
    """
    pretrained = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    compatible = {k: v for k, v in pretrained.items()
                  if k in model_dict and model_dict[k].shape == v.shape}
    skipped = [k for k in pretrained.keys() if k not in compatible]

    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(compatible)}/{len(pretrained)} tensors from checkpoint.")
    if skipped:
        print(f"Skipped (shape mismatch, likely classifier head): {skipped}")
    return model

def preflight_check_vehiclex_val_ids(xml_path, val_ids, pitch_min, pitch_max, min_images_per_id=8):
    """
    Counts images-per-identity (within the pitch-filtered pool) for the
    VehicleX val split, and warns if any identity is too sparse for a
    non-trivial query/gallery split. Catches the same failure mode that
    previously produced a misleading low F1 from a random-sampling VehicleX
    eval attempt. Does not raise — just warns, so a sparse split doesn't
    silently invalidate results.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    items = root.find("Items")

    val_id_set = set(val_ids)
    counts = Counter()
    for item in items.findall("Item"):
        vid = item.get("vehicleID")
        if vid not in val_id_set:
            continue
        camDis = float(item.get("camDis"))
        camHei = float(item.get("camHei"))
        pitch = math.degrees(math.atan2(camHei, camDis))
        if pitch_min <= pitch <= pitch_max:
            counts[vid] += 1

    missing = [vid for vid in val_id_set if counts.get(vid, 0) == 0]
    low = [(vid, c) for vid, c in counts.items() if 0 < c < min_images_per_id]

    print(f"\n[VehicleX preflight] Val identities: {len(val_id_set)}")
    print(f"[VehicleX preflight] Val identities with 0 images in pitch range: {len(missing)}")
    print(f"[VehicleX preflight] Val identities with < {min_images_per_id} images: {len(low)}")
    if counts:
        vals = list(counts.values())
        print(f"[VehicleX preflight] Images/identity — min {min(vals)}, median {int(np.median(vals))}, max {max(vals)}")

    if missing or low:
        print("[VehicleX preflight] WARNING: some val identities are too sparse for a reliable "
              "query/gallery split. Consider re-running split_vehiclex_ids.py with a different "
              "seed, or filtering these identities out of val_ids before training.")

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='ReID model trainer')
    parser.add_argument('--config', default=None, help='Config Path')
    parser.add_argument('--batch_size', default=None, type=int, help='Batch size')
    parser.add_argument('--backbone', default=None, help='Model Backbone')
    parser.add_argument('--hflip', default=None, type=float, help='Probabilty for horizontal flip')
    parser.add_argument('--randomerase', default=None, type=float,  help='Probabilty for random erasing')
    parser.add_argument('--dataset', default=None, help='Choose one of [Veri776, VERIWILD, Market1501, VehicleID, VehicleX, CityFlow]')
    parser.add_argument('--imgsize_x', default=None, type=int, help='width image')
    parser.add_argument('--imgsize_y', default=None, type=int, help='height image')
    parser.add_argument('--num_instances', default=None, type=int, help='Number of images belonging to an ID inside of batch, the numbers of IDs is batch_size/num_instances')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--softmax_loss', default=None, help='The loss used for classification')
    parser.add_argument('--metric_loss', default=None, help='The loss used as metric loss')
    parser.add_argument("--triplet_margin", default=None, type=float, help='With margin>0 uses normal triplet loss. If margin<=0 or None Soft Margin Triplet Loss is used instead!')
    parser.add_argument('--optimizer', default=None, help='Adam or SGD')
    parser.add_argument('--initial_lr', default=None, type=float, help='Initial learning rate after warm-up')
    parser.add_argument('--lambda_ce', default=None, type=float, help='multiplier of the classification loss')
    parser.add_argument('--lambda_triplet', default=None, type=float, help='multiplier of the metric loss')

    parser.add_argument('--parallel', default=None, help='Whether to used DataParallel for multi-gpu in one device')    
    parser.add_argument('--half_precision', default=None, help='Use of mixed precision') 
    parser.add_argument('--mean_losses', default=None, help='Use of mixed precision') 
    parser.add_argument('--finetune_from', default=None, help='Optional path to a checkpoint to warm-start from (fine-tuning). Omit for training from scratch.')
    parser.add_argument('--eval_only', default=None, metavar='CHECKPOINT_PATH', help='Skip training entirely; load this checkpoint and run a single test_epoch against the selected dataset\'s query/gallery.')
    
    args = parser.parse_args()

    ### Load hyper parameters
    if args.config:
        with open(args.config, "r") as stream:
            data = yaml.safe_load(stream)
    else:
        with open("./config/config.yaml", "r") as stream:
            data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['p_hflip'] = args.hflip or data['p_hflip']
    data['y_length'] = args.imgsize_y or data['y_length']
    data['x_length'] = args.imgsize_x or data['x_length']
    data['p_rerase'] = args.randomerase or data['p_rerase']
    data['dataset'] = args.dataset or data['dataset']
    data['NUM_INSTANCES'] = args.num_instances or data['NUM_INSTANCES']
    data['model_arch'] = args.model_arch or data['model_arch']
    if args.triplet_margin is not None: data['triplet_margin'] = args.triplet_margin
    data['softmax_loss'] = args.softmax_loss or data['softmax_loss']
    data['metric_loss'] = args.metric_loss or data['metric_loss']
    data['optimizer'] = args.optimizer or data['optimizer']
    data['lr'] = args.initial_lr or data['lr']
    data['parallel'] = args.parallel or data['parallel']
    data['alpha_ce'] = args.lambda_ce or data['alpha_ce']
    data['beta_tri'] = args.lambda_triplet or data['beta_tri']
    # data['gamma_ce'] = args.gamma_ce or data['gamma_ce']
    # data['gamma_t'] = args.gamma_t or data['gamma_t']
    data['backbone'] = args.backbone or data['backbone']
    data['half_precision'] = args.half_precision or data['half_precision']
    if args.mean_losses is not None: data['mean_losses'] = bool(args.mean_losses)
    if args.finetune_from is not None: data['finetune_from'] = args.finetune_from


    alpha_ce= data['alpha_ce']
    beta_tri = data['beta_tri']

    #### Set Seed for consistent and deterministic results
    set_seed(data['torch_seed'])
    ### Config print
    print("\n\n\n  Config used: \n")
    print(data)
    print("\n\n\n End config")

    teste_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.Normalize(data['n_mean'], data['n_std']),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((data['y_length'], data['x_length']), antialias=True),
        transforms.Pad(10),
        transforms.RandomCrop((data['y_length'], data['x_length'])),
        transforms.RandomHorizontalFlip(p=data['p_hflip']),
        transforms.Normalize(data['n_mean'], data['n_std']),
        transforms.RandomErasing(p=data['p_rerase']),
    ])

    #### Dataset Loading       
    if data['dataset']== "VehicleID":
        data_q = CustomDataSet4VehicleID('/home/eurico/VehicleID_V1.0/train_test_split/test_list_800.txt', data['ROOT_DIR'], is_train=False, mode="q", transform=teste_transform)
        data_g = CustomDataSet4VehicleID('/home/eurico/VehicleID_V1.0/train_test_split/test_list_800.txt', data['ROOT_DIR'], is_train=False, mode="g", transform=teste_transform)
        data_train = CustomDataSet4VehicleID("/home/eurico/VehicleID_V1.0/train_test_split/train_list.txt", data['ROOT_DIR'], is_train=True, transform=train_transform)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)#
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
    if data['dataset']== 'VERIWILD':
        data_q = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id_query.txt', data['ROOT_DIR'], transform=teste_transform, with_view=False)
        data_g = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id.txt', data['ROOT_DIR'], transform=teste_transform, with_view=False)
        data_train = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/train_list.txt', data['ROOT_DIR'], transform=train_transform, with_view=False)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)#
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        if data["LAI"]:
            data_train = CustomDataSet4Veri776_withviewpont(data['train_list_file'], data['train_dir'], data['train_keypoint'], data['test_keypoint'], is_train=True, transform=train_transform)
        else:
            data_train = CustomDataSet4Veri776(data['train_list_file'], data['train_dir'], is_train=True, transform=train_transform)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'VehicleX':
        # Requires CustomDataSet4VehicleX.__init__ to set self.data_info = self.names
        # (RandomIdentitySampler expects .data_info; see conversation notes).
        with open("dataset/vehiclex_data/vehiclex_train_ids.txt") as f:
            train_ids = [l.strip() for l in f if l.strip()]
        with open("dataset/vehiclex_data/vehiclex_val_ids.txt") as f:
            val_ids = [l.strip() for l in f if l.strip()]

        pitch_min = data.get('vehiclex_pitch_min', 20)
        pitch_max = data.get('vehiclex_pitch_max', 60)

        if data.get('vehiclex_preflight', True):
            preflight_check_vehiclex_val_ids(
                data['train_list_file'], val_ids, pitch_min, pitch_max,
                min_images_per_id=data.get('vehiclex_min_images_per_val_id', 8))

        data_train = CustomDataSet4VehicleX(
            xml_path=data['train_list_file'], root_dir=data['train_dir'],
            is_train=True, transform=train_transform,
            pitch_min=pitch_min, pitch_max=pitch_max,
            n_view_bins=data['n_views'], id_list=train_ids)
        data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)

        # data_q/data_g point at VehicleX's own held-out query/gallery split
        # (not VeRi), so the training loop and eval call below need no
        # dataset-specific branching at all.
        data_q = CustomDataSet4VehicleX(
            xml_path=data['train_list_file'], root_dir=data['train_dir'],
            is_train=False, transform=teste_transform,
            pitch_min=pitch_min, pitch_max=pitch_max,
            n_view_bins=data['n_views'], id_list=val_ids, split='query')
        data_g = CustomDataSet4VehicleX(
            xml_path=data['train_list_file'], root_dir=data['train_dir'],
            is_train=False, transform=teste_transform,
            pitch_min=pitch_min, pitch_max=pitch_max,
            n_view_bins=data['n_views'], id_list=val_ids, split='gallery')
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset'] == 'CityFlow':
        # CityFlow ships its own official train/query/test XML partition
        # (train_label.xml / query_label.xml / test_label.xml) — unlike
        # VehicleX, no custom split is built here; using the official split
        # is what makes this number comparable to literature.
        # Filenames are sequential (000001.jpg...), NOT vehicleID-encoded
        # like VeRi-776 — that's why CustomDataSet4Veri776 can't be reused
        # here; CustomDataSet4CityFlow parses the XML directly instead.
        if args.eval_only:
            # Test-only run: never touch train_label.xml/image_train at all.
            data_train = None
        else:
            data_train = CustomDataSet4CityFlow(
                xml_path=data['train_list_file'], root_dir=data['train_dir'],
                is_train=True, transform=train_transform)
            data_train = DataLoader(data_train, sampler=RandomIdentitySampler(data_train, data['BATCH_SIZE'], data['NUM_INSTANCES']), num_workers=data['num_workers_train'], batch_size = data['BATCH_SIZE'], collate_fn=train_collate_fn, pin_memory=True)

        data_q = CustomDataSet4CityFlow(
            xml_path=data['query_list_file'], root_dir=data['query_dir'],
            is_train=False, transform=teste_transform)
        data_g = CustomDataSet4CityFlow(
            xml_path=data['gallery_list_file'], root_dir=data['teste_dir'],
            is_train=False, transform=teste_transform)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    # Check if the GPU is available and select
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Create Model
    model = get_model(data, device)
    if data.get('finetune_from'):
        print(f"Warm-starting from: {data['finetune_from']}")
        model = load_pretrained_backbone(model, data['finetune_from'], device)
    if data['parallel']:
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        print("\n \n Parallel activated!\nDo not use this with LBS!\nIt may result in weird behaviour sometimes.")

    if args.eval_only:
        # Skip training entirely: load the given checkpoint, run one
        # test_epoch against the dataset selected above, print, exit.
        # Note: --finetune_from and --eval_only both call
        # load_pretrained_backbone; if both are set, eval_only's checkpoint
        # loads second and wins.
        print(f"\n[eval_only] Loading checkpoint: {args.eval_only}")
        model = load_pretrained_backbone(model, args.eval_only, device)
        model.eval()
        eval_logger = Logger(data)
        # remove_junk assumption: True for any real multi-camera dataset
        # (VeRi-style same-camera junk filtering applies), False only for
        # VehicleX (synthetic, different camera/junk convention, established
        # earlier in this project). CityFlow is a real multi-camera network
        # dataset like VeRi, so it defaults to True here — flagged as an
        # assumption, not confirmed against CityFlow's own eval protocol.
        remove_junk = data['dataset'] not in ('VehicleX',)
        cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'],
                               eval_logger, epoch=0, remove_junk=remove_junk, scaler=False)
        print(f"\n[eval_only] {data['dataset']}  mAP {mAP:.4f}  CMC1 {cmc[0]:.4f}")
        eval_logger.save_log()
        sys.exit(0)

    ### Losses ###
    loss_fn = nn.CrossEntropyLoss(label_smoothing=data['label_smoothing'])
    # metric_loss: 'fastreidtriplet' (existing default, unchanged) or
    # 'supcon' (opt-in). Nothing else about this block changes — same
    # alpha_ce/beta_tri weighting applies to whichever loss object is built
    # here, since train_epoch just calls metric_loss(item, label) either way.
    if data.get('metric_loss') == 'supcon':
        print(f"Using SupConLoss (temperature={data.get('supcon_temperature', 0.07)}) in place of triplet loss.")
        metric_loss = SupConLoss(temperature=data.get('supcon_temperature', 0.07))
    else:
        metric_loss = triplet_loss_fastreid(data['triplet_margin'], norm_feat=data['triplet_norm'], hard_mining=data['hard_mining'])

    
    #### Optimizer
    optimizer = make_optimizer(data['optimizer'],
                            model,
                            data['lr'],
                            data['weight_decay'],
                            data['bias_lr_factor'],
                            data['momentum'])              #data['eps'])
    ### Schedule for the optimizer           
    if data['epoch_freeze_L1toL3'] == 0:                 
        scheduler = make_warmup_scheduler(data['sched_name'],
                                        optimizer,
                                        data['num_epochs'],
                                        data['milestones'],
                                        data['gamma'],
                                        data['warmup_factor'],
                                        data['warmup_iters'],
                                        data['warmup_method'],
                                        last_epoch=-1,
                                        min_lr = data['min_lr']
                                        )
    else:
        scheduler = None

    ### If running with fp16 precision
    if data['half_precision']:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler=False

    ### Initiate a Logger with TensorBoard to store Scalars, Embeddings and the weights of the model
    logger = Logger(data)

    ##freeze backbone at warmupup epochs up to data['warmup_iters'] 
    if data['freeze_backbone_warmup']:
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        for param in model.modelL4.parameters():
            param.requires_grad = False
    if data['epoch_freeze_L1toL3'] > 0:
        ### Freeze up to the penultimate layer    
        for param in model.modelup2L3.parameters():
            param.requires_grad = False
        print("\nFroze Backbone before branches!")
  

    ## Training Loop
    for epoch in tqdm(range(data['num_epochs'])):
        ##unfreeze backbone
        if epoch == data['warmup_iters'] -1: 
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            for param in model.modelL4.parameters():
                param.requires_grad = True

        if epoch == data['epoch_freeze_L1toL3']-1:
            scheduler = make_warmup_scheduler(data['sched_name'],
                                            optimizer,
                                            data['num_epochs'],
                                            data['milestones'],
                                            data['gamma'],
                                            data['warmup_factor'],
                                            data['warmup_iters'],
                                            data['warmup_method'],
                                            last_epoch=-1,
                                            min_lr = data['min_lr']
                                            )
            for param in model.modelup2L3.parameters():
                param.requires_grad = True
            print("\nUnfrozen Backbone before branches!")

        ###step schedule
        if epoch >= data['epoch_freeze_L1toL3']-1:              
            scheduler.step()    

        # None outside the freeze period / when epoch_freeze_L1toL3 isn't used —
        # requires processor.py's train_epoch to accept freeze_bn_modules=None
        # as a no-op default (see processor.py changes).
        freeze_modules = model.modelup2L3 if epoch < data['epoch_freeze_L1toL3'] - 1 else None
        train_loss, c_loss, t_loss, alpha_ce, beta_tri = train_epoch(
            model, device, data_train, loss_fn, metric_loss,
            optimizer, data, alpha_ce, beta_tri, logger, epoch,
            scheduler, scaler, freeze_bn_modules=freeze_modules
        )
        ###Evaluation
        if epoch%data['validation_period']==0 or epoch>=data['num_epochs']-15:
            # VeRi's junk-removal logic doesn't apply to VehicleX; every other
            # dataset here keeps the original remove_junk=True behavior.
            remove_junk = data['dataset'] != 'VehicleX'
            cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], logger, epoch, remove_junk=remove_junk, scaler=scaler)
            print(f'\n EPOCH {epoch+1}/{data["num_epochs"]} {data["dataset"]} mAP {mAP} CMC1 {cmc[0]}')

            logger.save_model(model, epoch=epoch)
    print("Best mAP: ", np.max(logger.logscalars['Accuraccy/mAP']))
    print("Best CMC1: ", np.max(logger.logscalars['Accuraccy/CMC1']))
    logger.save_log()