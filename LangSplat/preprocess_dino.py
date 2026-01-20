import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn


@dataclass
class DinoTextNetworkConfig:
    _target: Type = field(default_factory=lambda: DinoTextNetwork)
    dino_n_dims: int = 2048  # DinoText embedding dimension
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)


class DinoTextNetwork(nn.Module):
    def __init__(self, config: DinoTextNetworkConfig):
        super().__init__()
        self.config = config

        # Import DinoText model and tokenizer
        import sys
        sys.path.insert(0, '/home/s25mdeyl_hpc/workspace/dinov3')
        from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

        print("Loading DinoText model...")
        model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(pretrained=True)
        model.eval()
        self.model = model.to("cuda")
        self.tokenizer = tokenizer
        self.dino_n_dims = self.config.dino_n_dims

        # DinoV3 normalization (ImageNet stats)
        self.process = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.positives = self.config.positives
        self.negatives = self.config.negatives

        # Encode positive and negative text embeddings
        with torch.no_grad():
            pos_tokens = self.tokenizer.tokenize(list(self.positives)).to("cuda")
            self.pos_embeds = self.model.encode_text(pos_tokens, normalize=True)

            neg_tokens = self.tokenizer.tokenize(list(self.negatives)).to("cuda")
            self.neg_embeds = self.model.encode_text(neg_tokens, normalize=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.dino_n_dims
        ), f"Embedding dimensionality must match the model dimensionality: {self.pos_embeds.shape[1]} vs {self.dino_n_dims}"

    @property
    def name(self) -> str:
        return "dinotext_vitl16_tet1280d20h24l"

    @property
    def embedding_dim(self) -> int:
        return self.config.dino_n_dims

    def gui_cb(self, element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            pos_tokens = self.tokenizer.tokenize(list(self.positives)).to("cuda")
            self.pos_embeds = self.model.encode_text(pos_tokens, normalize=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 2048
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        """Encode images using DinoText vision encoder"""
        # Process and normalize input
        processed_input = self.process(input)

        # Get model dtype
        model_dtype = next(self.model.parameters()).dtype
        processed_input = processed_input.to(model_dtype)

        # Encode image - returns global CLS embedding (not patch tokens)
        with torch.no_grad():
            features = self.model.encode_image(processed_input, normalize=True)

        return features




def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images with DinoText", total=len(image_list)):
        # Check if this image is already processed
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        if os.path.exists(save_path + '_s.npy') and os.path.exists(save_path + '_f.npy'):
            continue

        try:
            img_embed, seg_map = _embed_dino_sam_tiles(img.unsqueeze(0), sam_encoder)
        except Exception as e:
            print(f"Error processing image {i} ({data_list[i]}): {e}")
            continue

        # Concatenate embeddings from all scales
        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        img_embed_concat = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed_concat.shape[0] == total_length

        # Process segmentation maps
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]

        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            if v.max() >= 0:  # Only process if there are valid segments
                assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
                v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))

        seg_map_stacked = torch.stack(seg_map_tensor, dim=0)

        # Save immediately after processing this image
        assert total_length == int(seg_map_stacked.max() + 1), f"Mismatch: {total_length} vs {int(seg_map_stacked.max() + 1)}"

        curr = {
            'feature': img_embed_concat,
            'seg_maps': seg_map_stacked
        }
        sava_numpy(save_path, curr)

    mask_generator.predictor.model.to('cpu')


def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())


def _embed_dino_sam_tiles(image, sam_encoder):
    """Embed SAM segments using DinoText"""
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    dino_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        if mode not in seg_images:
            continue

        tiles = seg_images[mode]
        tiles = tiles.to("cuda")

        with torch.no_grad():
            dino_embed = model.encode_image(tiles)

        # Normalize embeddings
        dino_embed = dino_embed / dino_embed.norm(dim=-1, keepdim=True)
        dino_embeds[mode] = dino_embed.detach().cpu().half()

    return dino_embeds, seg_map


def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img


def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad


def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep


def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]

    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx


def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new


def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    print("Initializing DinoText model...")
    model = DinoTextNetwork(DinoTextNetworkConfig)
    print(f"DinoText model loaded. Embedding dimension: {model.embedding_dim}")

    print("Initializing SAM model...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_list = []
    WARNED = False
    print("Loading images...")
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'language_features_dino')
    os.makedirs(save_folder, exist_ok=True)
    print(f"Generating features and saving to {save_folder}...")
    create(imgs, data_list, save_folder)
    print("Done!")
