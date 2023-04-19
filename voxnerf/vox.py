import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from my.registry import Registry

VOXRF_REGISTRY = Registry("VoxRF")


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs, embed_fr = 1.0):
        arr = []
        num_embed_fns = len(self.embed_fns)
        for i in range(num_embed_fns):
            if i < num_embed_fns*embed_fr or i == 0:
                arr.append(self.embed_fns[i](inputs))
            else:
                arr.append(torch.zeros_like(arr[-1]))
        return torch.cat(arr, -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, embed_fr, eo=embedder_obj : eo.embed(x, embed_fr)
    return embed, embedder_obj.out_dim



class VanillaNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[]):
        """ 
        """
        super(VanillaNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
    
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        print("Model input shape: " + str(x.shape))
        print("Input channels: " + str(self.input_ch))
        input_pts = x
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)

        return outputs

    def opt_params(self):
        groups = []
        print("Set opt params")
        for name, param in self.named_parameters():
            grp = {"params": param, "lr": 5e-4}
            groups.append(grp)
    
        return groups

    def annealed_opt_params(self, base_lr, σ):
        groups = []
        print("Annealed opt params")
        for name, param in self.named_parameters():
            grp = {"params": param, "lr": 5e-4 * σ}
            groups.append(grp)
            
        return groups   



def to_grid_samp_coords(xyz_sampled, aabb):
    # output range is [-1, 1]
    aabbSize = aabb[1] - aabb[0]
    return (xyz_sampled - aabb[0]) / aabbSize * 2 - 1


def add_non_state_tsr(nn_module, key, val):
    # tsr added here does not appear in module's state_dict;
    nn_module.register_buffer(key, val, persistent=False)


@VOXRF_REGISTRY.register()
class VoxRF(nn.Module):
    def __init__(
        self, aabb, grid_size, step_ratio=0.5,
        density_shift=-10, ray_march_weight_thres=0.0001, c=3,
        blend_bg_texture=True, bg_texture_hw=64
    ):
        assert aabb.shape == (2, 3)
        xyz = grid_size
        del grid_size

        super().__init__()
        add_non_state_tsr(self, "aabb", torch.tensor(aabb, dtype=torch.float32))
        add_non_state_tsr(self, "grid_size", torch.LongTensor(xyz))

        self.density_shift = density_shift
        self.ray_march_weight_thres = ray_march_weight_thres
        self.step_ratio = step_ratio

        zyx = xyz[::-1]
        self.density = torch.nn.Parameter(
            torch.zeros((1, 1, *zyx))
        )
        self.color = torch.nn.Parameter(
            torch.randn((1, c, *zyx))
        )

        self.blend_bg_texture = blend_bg_texture
        self.bg = torch.nn.Parameter(
            torch.randn((1, c, bg_texture_hw, bg_texture_hw))
        )

        self.c = c
        self.alphaMask = None
        self.feats2color = lambda feats: torch.sigmoid(feats)

        self.d_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.embed_fn, self.embed_dim = get_embedder(10, 0)
        self.app_net = VanillaNeRF(input_ch=self.embed_dim, output_ch=4, D=6, W=196, skips=[])

    @property
    def device(self):
        return self.density.device

    def compute_density_feats(self, xyz_sampled):
        xyz_sampled = to_grid_samp_coords(xyz_sampled, self.aabb)
        n = xyz_sampled.shape[0]
        xyz_sampled = xyz_sampled.reshape(1, n, 1, 1, 3)
        σ = F.grid_sample(self.density, xyz_sampled).view(n)
        # We notice that DreamFusion also uses an exp scaling on densities.
        # The technique here is developed BEFORE DreamFusion came out,
        # and forms part of our upcoming technical report discussing invariant
        # scaling for volume rendering. The reseach was presented to our
        # funding agency (TRI) on Aug. 25th, and discussed with a few researcher friends
        # during the period.
        σ = σ * torch.exp(self.d_scale)
        σ = F.softplus(σ + self.density_shift)
        return σ

    def compute_app_feats_vanilla(self, xyz_sampled, xyz_weights, embed_fr=1.0):
        input = xyz_sampled#torch.cat((xyz_sampled, xyz_weights.unsqueeze(-1)), -1)
        input = self.embed_fn(input, embed_fr=embed_fr)
        feats = self.app_net(input)
        return feats

    def compute_bg(self, uv):
        n = uv.shape[0]
        uv = uv.reshape(1, n, 1, 2)
        feats = F.grid_sample(self.bg, uv).view(self.c, n)
        feats = feats.T
        return feats

    def get_per_voxel_length(self):
        aabb_size = self.aabb[1] - self.aabb[0]
        # NOTE I am not -1 on grid_size here;
        # I interpret a voxel as a square and val sits at the center; like pixel
        # this is consistent with align_corners=False
        vox_xyz_length = aabb_size / self.grid_size
        return vox_xyz_length

    def get_num_samples(self, max_size=None):
        # funny way to set step size; whatever
        unit = torch.mean(self.get_per_voxel_length())
        step_size = unit * self.step_ratio
        step_size = step_size.item()  # get the float

        if max_size is None:
            aabb_size = self.aabb[1] - self.aabb[0]
            aabb_diag = torch.norm(aabb_size)
            max_size = aabb_diag

        num_samples = int((max_size / step_size).item()) + 1
        return num_samples, step_size

    @torch.no_grad()
    def resample(self, target_xyz: list):
        zyx = target_xyz[::-1]
        self.density = self._resamp_param(self.density, zyx)
        self.color = self._resamp_param(self.color, zyx)
        target_xyz = torch.LongTensor(target_xyz).to(self.aabb.device)
        add_non_state_tsr(self, "grid_size", target_xyz)

    @staticmethod
    def _resamp_param(param, target_size):
        return torch.nn.Parameter(F.interpolate(
            param.data, size=target_size, mode="trilinear"
        ))

    @torch.no_grad()
    def compute_volume_alpha(self):
        xyz = self.grid_size.tolist()
        unit_xyz = self.get_per_voxel_length()
        xs, ys, zs = torch.meshgrid(
            *[torch.arange(nd) for nd in xyz], indexing="ij"
        )
        pts = torch.stack([xs, ys, zs], dim=-1).to(unit_xyz.device)  # [nx, ny, nz, 3]
        pts = self.aabb[0] + (pts + 0.5) * unit_xyz
        pts = pts.reshape(-1, 3)
        # could potentially filter with alpha mask itself if exists
        σ = self.compute_density_feats(pts)
        d = torch.mean(unit_xyz)
        α = 1 - torch.exp(-σ * d)
        α = rearrange(α.view(xyz), "x y z -> 1 1 z y x")
        α = α.contiguous()
        return α

    @torch.no_grad()
    def make_alpha_mask(self):
        α = self.compute_volume_alpha()
        ks = 3
        α = F.max_pool3d(α, kernel_size=ks, padding=ks // 2, stride=1)
        α = (α > 0.08).float()
        vol_mask = AlphaMask(self.aabb, α)
        self.alphaMask = vol_mask

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.alphaMask is not None:
            state['alpha_mask'] = self.alphaMask.export_state()
        return state

    def load_state_dict(self, state_dict):
        if 'alpha_mask' in state_dict.keys():
            state = state_dict.pop("alpha_mask")
            self.alphaMask = AlphaMask.from_state(state)
        return super().load_state_dict(state_dict, strict=True)


@VOXRF_REGISTRY.register()
class V_SJC(VoxRF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rendering color in [-1, 1] range, since score models all operate on centered img
        self.feats2color = lambda feats: torch.sigmoid(feats) * 2 - 1

    def opt_params(self):
        groups = []
        for name, param in self.named_parameters():
            # print(f"{name} {param.shape}")
            grp = {"params": param}
            if name in ["bg"]:
                grp["lr"] = 0.0001
            if name in ["density"]:
                # grp["lr"] = 0.
                pass
            if "app_net" in name:
                print("Initializing learning rate for app_net to 5e-4")
                grp["lr"] = 5e-3
            groups.append(grp)
        return groups

    def annealed_opt_params(self, base_lr, σ):
        groups = []
        for name, param in self.named_parameters():
            # print(f"{name} {param.shape}")
            grp = {"params": param, "lr": base_lr * σ}
            if name in ["density"]:
                grp["lr"] = base_lr * σ
            if name in ["d_scale"]:
                grp["lr"] = 0.
            if name in ["color"]:
                grp["lr"] = base_lr * σ
            if "app_net" in name:
                grp["lr"] = 5e-4 * σ
                print("Anneling app_net")
            if name in ["bg"]:
                grp["lr"] = 0.01
            groups.append(grp)
        return groups


@VOXRF_REGISTRY.register()
class V_SD(V_SJC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # rendering in feature space; no sigmoid thresholding
        self.feats2color = lambda feats: feats


class AlphaMask(nn.Module):
    def __init__(self, aabb, alphas):
        super().__init__()
        zyx = list(alphas.shape[-3:])
        add_non_state_tsr(self, "alphas", alphas.view(1, 1, *zyx))
        xyz = zyx[::-1]
        add_non_state_tsr(self, "grid_size", torch.LongTensor(xyz))
        add_non_state_tsr(self, "aabb", aabb)

    def sample_alpha(self, xyz_pts):
        xyz_pts = to_grid_samp_coords(xyz_pts, self.aabb)
        xyz_pts = xyz_pts.view(1, -1, 1, 1, 3)
        α = F.grid_sample(self.alphas, xyz_pts).view(-1)
        return α

    def export_state(self):
        state = {}
        alphas = self.alphas.bool().cpu().numpy()
        state['shape'] = alphas.shape
        state['mask'] = np.packbits(alphas.reshape(-1))
        state['aabb'] = self.aabb.cpu()
        return state

    @classmethod
    def from_state(cls, state):
        shape = state['shape']
        mask = state['mask']
        aabb = state['aabb']

        length = np.prod(shape)
        alphas = torch.from_numpy(
            np.unpackbits(mask)[:length].reshape(shape)
        )
        amask = cls(aabb, alphas.float())
        return amask


def test():
    device = torch.device("cuda:1")

    aabb = 1.5 * np.array([
        [-1, -1, -1],
        [1, 1, 1]
    ])
    model = VoxRF(aabb, [10, 20, 30])
    model.to(device)
    print(model.density.shape)
    print(model.grid_size)

    return


if __name__ == "__main__":
    test()
