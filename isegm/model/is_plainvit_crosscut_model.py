import jittor as jt
import jittor.nn as nn
import numpy as np
from isegm.model.ops import BatchImageNormalize, ScaleLayer


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks

    def get_coord_features(self, points, batchsize, rows, cols, n):
        if self.cpu_mode:
            print('please use gpu mode')
            exit()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = jt.split(points, [2, 1], dim=1)

            invalid_points = jt.max(points, dim=1, keepdim=False) < 0
            row_array = jt.arange(start=0, end=rows, step=1, dtype=jt.float32)
            col_array = jt.arange(start=0, end=cols, step=1, dtype=jt.float32)

            coord_rows, coord_cols = jt.meshgrid(row_array, col_array)
            coords = jt.stack([coord_rows, coord_cols], dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)

            coords -= add_xy
            if not self.use_disks:
                coords /= (self.norm_radius * self.spatial_scale)
            coords *= coords  # 或 coords = coords * coords
            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6
            coords = coords.view(-1, num_points, 1, rows, cols)  #2B points 1 h w

            ##########【ZN】##########
            distance_map = coords.clone()  # 2B points 1 h w
            min_indices = distance_map.argmin(dim=1)[0]  # 2B 1 h w
            min_indices = min_indices.view(-1, 2, rows, cols)  # B 2 h w

            distance_map = distance_map.squeeze(2)  # 2B points h w
            seperate_coords = distance_map.view(-1, 2, num_points, rows, cols)  # B 2 points h w
            seperate_coords = (seperate_coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
            #-------------------------#
            coords = coords.min(dim=1)  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)  # B 2 h w

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        ##########【ZN】##########
        new_H, new_W = coords.shape[2] // n, coords.shape[3] // n
        B = coords.shape[0]
        coords = jt.nn.unfold(coords, kernel_size=(new_H, new_W), stride=(new_H, new_W)).reshape(B, 2, new_H,
                                                                                                 new_W, -1).permute(0,
                                                                                                                    4,
                                                                                                                    1,
                                                                                                                    2,
                                                                                                                    3)
        min_indices = jt.nn.unfold(min_indices, kernel_size=(new_H, new_W), stride=(new_H, new_W)).reshape(B, 2, new_H,
                                                                                                           new_W,
                                                                                                           -1).permute(
            0, 4, 1, 2, 3)

        #-------------------------#

        return coords, min_indices, seperate_coords

    def execute(self, x, coords, n):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3], n)


class ISModel(nn.Module):
    def __init__(self, with_aux_output=False, norm_radius=5, use_disks=False, cpu_dist_maps=False,
                 use_rgb_conv=False, use_leaky_relu=False,  # the two arguments only used for RITM
                 with_prev_mask=False, norm_mean_std=([.485, .456, .406], [.229, .224, .225]), slice_number=2):
        super().__init__()

        self.with_aux_output = with_aux_output
        self.with_prev_mask = with_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.slice_number = slice_number
        self.coord_feature_ch = 2
        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            # Only RITM models need to transform the coordinate features, though they don't use 
            # exact 'rgb_conv'. We keep 'use_rgb_conv' only for compatible issues.
            # The simpleclick models use a patch embedding layer instead 
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)
        else:
            self.maps_transform = nn.Identity()

        self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                  cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def execute(self, image, points, slice_number=None):

        image, prev_mask = self.prepare_input(image)
        if slice_number is not None:
            self.slice_number = slice_number
        orig_image = image.clone()

        n = self.slice_number

        target_size = 448 * n
        _, _, H, W = image.shape
        if H != target_size or W != target_size:
            image = nn.interpolate(image, size=(target_size, target_size), mode='bilinear', align_corners=False)
            prev_mask = nn.interpolate(prev_mask, size=(target_size, target_size), mode='nearest')
            points = self.resize_valid_points(points, (H, W), (target_size, target_size))

        inter_image = image.clone()

        B, C, new_H, new_W = image.shape[0], image.shape[1], image.shape[2] // n, image.shape[3] // n

        image = jt.nn.unfold(image, kernel_size=(new_H, new_W), stride=(new_H, new_W)).reshape(B, C,
                                                                                               new_H, new_W,
                                                                                               n * n).permute(0, 4, 1,
                                                                                                              2, 3)

        prev_mask = jt.nn.unfold(prev_mask, kernel_size=(new_H, new_W), stride=(new_H, new_W)).reshape(B, 1, new_H,
                                                                                                       new_W,
                                                                                                       n * n).permute(0,
                                                                                                                      4,
                                                                                                                      1,
                                                                                                                      2,
                                                                                                                      3)

        coord_features, min_indices, seperate_coords = self.get_coord_features(inter_image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)
        outputs = self.backbone_forward(image, coord_features, min_indices, seperate_coords)

        outputs['instances'] = nn.interpolate(outputs['instances'], size=orig_image.size()[2:],
                                              mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.interpolate(outputs['instances_aux'], size=orig_image.size()[2:],
                                                      mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]

        image = self.normalization(image)
        return image, prev_mask

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        coord_features, min_indices, seperate_coords = self.dist_maps(image, points, self.slice_number)
        if prev_mask is not None:
            if prev_mask.dim() == 4:
                coord_features = jt.concat((prev_mask, coord_features), dim=1)
            elif prev_mask.dim() == 5:
                coord_features = jt.concat((prev_mask, coord_features), dim=2)

        return coord_features, min_indices, seperate_coords

    def resize_valid_points(self, points, old_size, new_size, align_corners=False):

        H_old, W_old = old_size
        H_new, W_new = new_size

        if align_corners:
            scale_y = (H_new - 1) / (H_old - 1) if H_old > 1 else 1.0
            scale_x = (W_new - 1) / (W_old - 1) if W_old > 1 else 1.0
        else:
            scale_y = H_new / H_old
            scale_x = W_new / W_old

        points = points.clone().float()

        valid_mask = (points != -1).all(dim=-1)  # (B, N)

        points_y = points[..., 0]
        points_x = points[..., 1]

        points_y[valid_mask] = points_y[valid_mask] * scale_y
        points_x[valid_mask] = points_x[valid_mask] * scale_x

        points[..., 0] = points_y
        points[..., 1] = points_x

        return points


def split_points_by_order(tpoints: jt.array, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [jt.array(x, dtype=tpoints.dtype)
                    for x in group_points]

    return group_points


import math
from isegm.utils.serialization import serialize
from .modeling.models_vit import VisionTransformer, PatchEmbed
from .modeling.swin_transformer import SwinTransfomerSegHead


class SimpleFPN(nn.Module):
    def __init__(self, in_dim=768, out_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.down_4_chan = max(out_dims[0] * 2, in_dim // 2)
        self.down_4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_4_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan),
            nn.GELU(),
            nn.ConvTranspose2d(self.down_4_chan, self.down_4_chan // 2, 2, stride=2),
            nn.GroupNorm(1, self.down_4_chan // 2),
            nn.Conv2d(self.down_4_chan // 2, out_dims[0], 1),
            nn.GroupNorm(1, out_dims[0]),
            nn.GELU()
        )
        self.down_8_chan = max(out_dims[1], in_dim // 2)
        self.down_8 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, self.down_8_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_8_chan),
            nn.Conv2d(self.down_8_chan, out_dims[1], 1),
            nn.GroupNorm(1, out_dims[1]),
            nn.GELU()
        )
        self.down_16 = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[2], 1),
            nn.GroupNorm(1, out_dims[2]),
            nn.GELU()
        )
        self.down_32_chan = max(out_dims[3], in_dim * 2)
        self.down_32 = nn.Sequential(
            nn.Conv2d(in_dim, self.down_32_chan, 2, stride=2),
            nn.GroupNorm(1, self.down_32_chan),
            nn.Conv2d(self.down_32_chan, out_dims[3], 1),
            nn.GroupNorm(1, out_dims[3]),
            nn.GELU()
        )

        self.init_weights()

    def init_weights(self):
        pass

    def execute(self, x):
        x_down_4 = self.down_4(x)
        x_down_8 = self.down_8(x)
        x_down_16 = self.down_16(x)
        x_down_32 = self.down_32(x)

        return [x_down_4, x_down_8, x_down_16, x_down_32]


class PlainVitCrosscutModel(ISModel):
    @serialize
    def __init__(
            self,
            backbone_params={},
            neck_params={},
            head_params={},
            random_split=False,
            **kwargs
    ):

        super().__init__(**kwargs)
        self.random_split = random_split

        self.patch_embed_coords = PatchEmbed(
            img_size=backbone_params['img_size'],
            patch_size=backbone_params['patch_size'],
            in_chans=3 if self.with_prev_mask else 2,
            embed_dim=backbone_params['embed_dim'],
        )

        self.backbone = VisionTransformer(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head = SwinTransfomerSegHead(**head_params)

        self.mlp_pos = nn.Sequential(
            nn.Linear(backbone_params['embed_dim'], backbone_params['embed_dim']),
            nn.ReLU(),
            nn.Linear(backbone_params['embed_dim'], backbone_params['embed_dim'])
        )

    def backbone_forward(self, image, coord_features=None, nearest_click_idx_map=None, each_click_map=None):

        C, H, W = image.shape[2], image.shape[3], image.shape[4]
        n = math.floor(math.sqrt(image.shape[1]))
        image_patches = image.view(-1, C, H, W)

        with jt.no_grad():
            points_features = self.backbone.forward_backbone(image_patches, None, self.random_split)
            B, N, C = points_features.shape
            grid_size = self.backbone.patch_embed.grid_size
            points_features = points_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])

            points_features = (
                points_features.view(-1, n, n, C, grid_size[0], grid_size[1])  # (B, n, n, C, h, w)
                .permute(0, 3, 1, 4, 2, 5)  # (B, C, n*h, n*w)
                .reshape(-1, C, grid_size[0] * n, grid_size[1] * n)
            )

            points_features = nn.interpolate(points_features,
                                             size=(int(grid_size[0] * 16 * n), int(grid_size[1] * 16 * n)),
                                             mode='bilinear')  # B C H W

            pos_click_map = each_click_map[:, 0, :, :, :]  # B,P,H,W
            positive_feature_list = []
            for i in range(each_click_map.shape[2]):
                positive_feature_list.append(
                    self.get_mean_feature(points_features, pos_click_map[:, i:i + 1, :, :]))  ## B P W H
            positive_feature = jt.stack(positive_feature_list).repeat(1, n * n, 1)  # P B*S C

            nearest_click_idx_map = nearest_click_idx_map.view(-1, 2, H, W)  # B 2 H W
            pos_nearest_click_idx_map = nearest_click_idx_map[:, 0, :, :]  # B H W

            B, H, W = pos_nearest_click_idx_map.shape
            P, BS, C = positive_feature.shape

            positive_feature = positive_feature.permute(1, 0, 2)  # (B*S, P, C)
            positive_feature_out = []

            for i in range(BS):
                idx_map = pos_nearest_click_idx_map[i]
                feat_bank = positive_feature[i]  # (P, C)
                selected = feat_bank[idx_map].permute(2, 0, 1).unsqueeze(0)
                selected = nn.interpolate(selected, size=(grid_size[0], grid_size[1]), mode='bilinear').squeeze(
                    0).permute(1, 2, 0)
                positive_feature_out.append(selected)

            positive_feature = jt.stack(positive_feature_out).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)

            pos_feature = positive_feature.detach().clone()

        coord_features = coord_features.view(-1, 3, H, W)
        coord_features = self.patch_embed_coords(coord_features)
        coord_features += self.mlp_pos(pos_feature)

        backbone_features = self.backbone.forward_backbone(image_patches, coord_features, self.random_split)

        # Extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = backbone_features.shape
        grid_size = self.backbone.patch_embed.grid_size

        backbone_features = backbone_features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = self.neck(backbone_features)

        instances = self.head(multi_scale_features)
        C, H, W = instances.shape[1], instances.shape[2], instances.shape[3]
        instances = instances.view(-1, n, n, C, H, W).permute(0, 3, 1, 4, 2, 5).reshape(-1, C, H * n, W * n)

        return {'instances': instances, 'instances_aux': None}

    def get_mean_feature(self, feature, mask):

        m = mask.float()

        sum_feat = (feature * m).sum(dims=(2, 3))

        counts = m.sum(dims=(2, 3)).squeeze(1)

        den = counts.unsqueeze(1)
        den = jt.where(den == 0, jt.ones_like(den), den)

        mean = sum_feat / den

        return mean  # (B, C)

