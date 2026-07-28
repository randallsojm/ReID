"""
inference_fixed.py
------------------
Fixed version of inference.py with per-branch L2 normalisation.

KEY FIX:
    Original inference.py:
        embedding = torch.cat(ffs, dim=1)      # concat first
        embedding = F.normalize(embedding, 1)  # normalise whole 8192-dim vector

    Official evaluation script (test_epoch):
        for item in ffs:
            end_vec.append(F.normalize(item))  # normalise each 2048-dim branch
        embedding = torch.cat(end_vec, 1)      # then concat

    The difference: per-branch normalisation ensures each branch contributes
    equally to the final cosine similarity, regardless of magnitude differences
    between branches. This matches how the model was trained and evaluated.

Everything else is identical to inference.py.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json


# ── Preprocessing ──────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])


def preprocess(image):
    """
    Convert any image format to a (1, 3, 256, 256) float32 tensor.

    Accepts:
        str        — file path
        np.ndarray — OpenCV BGR array
        PIL.Image  — PIL image

    Returns:
        torch.Tensor of shape (1, 3, 256, 256)
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image[:, :, ::-1])
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    else:
        raise TypeError(f"Unsupported image type: {type(image)}\n"
                        f"Pass a file path (str), PIL Image, or numpy BGR array.")
    return TRANSFORM(img).unsqueeze(0)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(weights_path, model_info_path=None, device=None):
    from models.models import MBR_model

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if model_info_path and os.path.exists(model_info_path):
        with open(model_info_path) as f:
            info = json.load(f)
        print(f"Loaded model info: arch={info['model_arch']}, "
              f"embedding_dim={info['embedding_dim']}, "
              f"trained for {info['total_epochs']} epochs")

    state_dict = torch.load(weights_path, map_location=device)
    first_key  = list(state_dict.keys())[0]
    if first_key.startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Infer class_num directly from the checkpoint's classifier shape,
    # instead of hardcoding 575 — makes this loader work for any checkpoint
    # (VeRi-trained, VehicleX-finetuned, etc.) since the classifier is
    # never used for embedding extraction anyway.
    classifier_key = 'finalblock.finalblocks.0.classifier.0.weight'
    inferred_class_num = state_dict[classifier_key].shape[0] if classifier_key in state_dict else 575
    print(f"Inferred class_num from checkpoint: {inferred_class_num}")

    model = MBR_model(
        class_num      = inferred_class_num,
        n_branches     = ["R50", "R50", "BoT", "BoT"],
        n_groups       = 0,
        losses         = "LBS",
        backbone       = "ibn",
        droprate       = 0,
        linear_num     = False,
        return_f       = True,
        circle_softmax = False,
        pretrain_ongroups = True,
        LAI            = False,
        n_cams         = 0,
        n_views        = 0,
    )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model loaded and ready.")
    return model, device


# ── Embedding extraction ───────────────────────────────────────────────────────

def extract_embedding(image, model, device):
    """
    Extract a per-branch normalised embedding from one image.

    FIX vs original inference.py:
        Each branch embedding (2048-dim) is L2-normalised individually
        before concatenation, matching the official test_epoch evaluation:

            for item in ffs:
                end_vec.append(F.normalize(item))   # per-branch norm
            embedding = torch.cat(end_vec, 1)       # (1, 8192)

        This ensures all 4 branches contribute equally to cosine similarity
        regardless of their raw magnitude differences.

    Args:
        image:  file path (str), PIL Image, or numpy BGR array
        model:  loaded MBR_model in eval mode
        device: torch.device

    Returns:
        torch.Tensor of shape (1, 8192), per-branch L2 normalised, on CPU
    """
    x = preprocess(image).to(device)

    cam  = torch.zeros(1, 1, dtype=torch.long).to(device)
    view = torch.zeros(1, 8, dtype=torch.long).to(device)

    with torch.no_grad():
        preds, embs, ffs, output = model(x, cam, view)

    # FIX: normalise each branch separately, then concatenate
    # Original: F.normalize(torch.cat(ffs, dim=1), dim=1)
    branch_embs = [F.normalize(ff, dim=1) for ff in ffs]
    embedding   = torch.cat(branch_embs, dim=1)   # (1, 8192), norm=2.0
    embedding   = F.normalize(embedding, dim=1)   # final unit norm → (1, 8192)

    return embedding.cpu()


# ── Gallery building ───────────────────────────────────────────────────────────

def build_gallery(image_dir, model, device, save_path=None,
                  extensions=('.jpg', '.png', '.jpeg')):
    """
    Extract embeddings for all images in a folder and optionally save.
    """
    image_files = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(extensions)
        and os.path.isfile(os.path.join(image_dir, f))
    ]
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Building gallery from {len(image_files)} images...")
    embeddings, paths, failed = [], [], []

    for i, path in enumerate(image_files):
        try:
            emb = extract_embedding(path, model, device)
            embeddings.append(emb)
            paths.append(path)
        except Exception as e:
            print(f"  WARNING: {path}: {e}")
            failed.append(path)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(image_files)} done...")

    gallery_embeddings = torch.cat(embeddings, dim=0)
    print(f"Gallery: {gallery_embeddings.shape[0]} images, "
          f"dim={gallery_embeddings.shape[1]}, failed={len(failed)}")

    if save_path:
        torch.save({'embeddings': gallery_embeddings, 'paths': paths}, save_path)
        print(f"Saved to {save_path}")

    return gallery_embeddings, paths


def load_gallery(gallery_path):
    data = torch.load(gallery_path, map_location='cpu')
    print(f"Gallery loaded: {data['embeddings'].shape[0]} images")
    return data['embeddings'], data['paths']


# ── Matching ───────────────────────────────────────────────────────────────────

def find_matches(query_image, model, device, gallery_embeddings,
                 gallery_paths, top_k=5):
    """
    Find top-k most similar gallery images for a query image.
    """
    query_emb = extract_embedding(query_image, model, device)
    scores    = torch.mm(query_emb, gallery_embeddings.t()).squeeze(0)
    top_scores, top_indices = scores.topk(min(top_k, len(gallery_paths)))

    return [
        {
            'rank':  rank + 1,
            'path':  gallery_paths[idx.item()],
            'score': round(score.item(), 4),
        }
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices))
    ]