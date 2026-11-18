"""The VAE depth encoders (streamvggt.depth_cond.conditioner.VAEDepthEncoder,
SD2 and FLUX.2): shapes, the zero-init no-op, which parameters train, the
latent against diffusers' own loader, and reload without the files. Needs
diffusers (the overlay) and the VAE weights; skipped without them."""

import os

import pytest
import torch

from streamvggt.depth_cond.conditioner import DepthConditioner, VAEDepthEncoder
from streamvggt.depth_cond.config import (
    DepthCondCfg,
    EncoderType,
    HeadType,
    InjectionType,
)

_CK = "/lustre/isaac24/proj/UTK0516/metrics_data/checkpoints_jd"
SD2_VAE = os.environ.get("SD2_VAE", f"{_CK}/sd2_vae")
FLUX2_VAE = os.environ.get("FLUX2_VAE", f"{_CK}/flux2_vae_mirror")
KINDS = {EncoderType.SD2_VAE: SD2_VAE, EncoderType.FLUX2_VAE: FLUX2_VAE}
pytest.importorskip("diffusers")
for _p in KINDS.values():
    if not os.path.isfile(os.path.join(_p, "diffusion_pytorch_model.safetensors")):
        pytest.skip(f"no VAE at {_p}", allow_module_level=True)

kinds = pytest.mark.parametrize("encoder", list(KINDS))


def _cfg(
    injection=InjectionType.TOKEN, encoder=EncoderType.SD2_VAE, **kw
) -> DepthCondCfg:
    cfg = DepthCondCfg(
        enabled=True,
        encoder=encoder,
        injection=injection,
        vae_path=KINDS[encoder],
        **kw,
    )
    cfg.validate()
    return cfg


def _cond(
    injection=InjectionType.TOKEN, out_spec=None, encoder=EncoderType.SD2_VAE
) -> DepthConditioner:
    return DepthConditioner(
        _cfg(injection, encoder), out_spec or {"token_dim": 32}, patch_size=14
    )


def _hint(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    depth = torch.rand(1, 2, 28, 42, generator=g) * 5 + 0.5
    mask = torch.rand(1, 2, 28, 42, generator=g) < 0.05
    return depth, mask


@kinds
def test_fresh_run_digests_the_weights_and_a_reload_needs_no_files(encoder) -> None:
    cfg = _cfg(encoder=encoder)
    assert len(cfg.vae_sha256) == 64
    with pytest.raises(ValueError, match="vae-path"):
        DepthCondCfg(
            enabled=True, encoder=encoder, injection=InjectionType.TOKEN
        ).validate()
    # a config back from a checkpoint carries the digest; the path may be gone
    reload = DepthCondCfg(
        enabled=True,
        encoder=encoder,
        injection=InjectionType.TOKEN,
        vae_path="",
        vae_sha256=cfg.vae_sha256,
    )
    reload.validate()
    a = DepthConditioner(cfg, {"token_dim": 32}, patch_size=14)
    b = DepthConditioner(reload, {"token_dim": 32}, patch_size=14)
    b.load_state_dict(a.state_dict(), strict=True)
    torch.nn.init.normal_(a.token_proj.weight)
    b.token_proj.load_state_dict(a.token_proj.state_dict())
    depth, mask = _hint(2)
    assert torch.allclose(a(depth, mask), b(depth, mask))
    # and the wrong VAE file is refused by a config that remembers another
    with pytest.raises(ValueError, match="sha256"):
        DepthCondCfg(
            enabled=True,
            encoder=encoder,
            injection=InjectionType.TOKEN,
            vae_path=KINDS[encoder],
            vae_sha256="0" * 64,
        ).validate()


@kinds
def test_token_arm_shapes_and_noop_at_init(encoder) -> None:
    cond = _cond(encoder=encoder)
    depth, mask = _hint()
    out = cond(depth, mask)
    assert out.shape == (1, 2, 2 * 3, 32)
    assert float(out.abs().max()) == 0.0
    torch.nn.init.normal_(cond.token_proj.weight)
    a = cond(depth, mask)
    b = cond(depth * 2, mask)
    assert not torch.allclose(a, b), "the output must depend on the hint"


@kinds
def test_vae_is_frozen_and_the_stem_trains(encoder) -> None:
    cond = _cond(encoder=encoder)
    frozen = {id(p) for p in cond.frozen_parameters()}
    assert frozen, "the VAE encoder's parameters must be reported frozen"
    for p in cond.parameters():
        if id(p) in frozen:
            assert not p.requires_grad
    # the recipe unfreezes the conditioner; frozen_parameters() is what the
    # model re-freezes afterwards (MetricStreamVGGT.freeze_for_finetune)
    enc = cond.encoder
    assert isinstance(enc, VAEDepthEncoder)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(("stem.", "refine.")) for n in trainable)
    # train() must not put the VAE encoder (or FLUX.2's bn) into train mode
    cond.train()
    assert not enc.vae_encoder.training
    if enc.bn is not None:
        assert not enc.bn.training
    # the frozen weights are part of the state dict (a reload needs no VAE files)
    assert any(k.startswith("encoder.vae_encoder.") for k in cond.state_dict())


@kinds
def test_latent_grid_is_two_cells_per_patch(encoder) -> None:
    enc = _cond(encoder=encoder).encoder
    filled = torch.rand(3, 28 * 2, 42 * 2) * 3  # transformed disparity, 56 x 84
    z = enc._latent(filled, 4, 6)
    assert z.shape == (3, enc.latent_channels * 4, 4, 6)
    # the fixed affine keeps metric scale: a frame twice as far changes the latent
    assert not torch.allclose(enc._latent(filled / 2, 4, 6), z)


def test_head_arm_shapes() -> None:
    spec = {"features": 8, "num_scales": 4, "heads": [HeadType.DEPTH]}
    cond = _cond(InjectionType.HEAD, spec)
    depth, mask = _hint(1)
    sizes = [(8, 12), (4, 6), (2, 3), (1, 2)]
    out = cond(depth, mask, out_hw_list=sizes)
    assert [tuple(r.shape[-2:]) for r in out["depth"]] == sizes


@kinds
def test_latent_matches_diffusers_from_pretrained(encoder) -> None:
    """The hand-loaded encoder (legacy attention keys renamed) must give
    diffusers' own encode() of the same directory, scaled as that VAE's
    pipeline feeds its transformer (SD2: scaling_factor; FLUX.2: the
    pipeline's _patchify_latents + BatchNorm running stats)."""
    import diffusers

    enc = _cond(encoder=encoder).encoder
    sd2 = encoder is EncoderType.SD2_VAE
    cls = diffusers.AutoencoderKL if sd2 else diffusers.AutoencoderKLFlux2
    ref = cls.from_pretrained(KINDS[encoder]).eval()
    g = torch.Generator().manual_seed(3)
    filled = torch.rand(1, 32, 48, generator=g) * 3
    img = (filled[:, None] / enc.VAE_INPUT_SCALE - 1.0).expand(-1, 3, -1, -1)
    with torch.no_grad():
        mean = ref.encode(img).latent_dist.mode()
        if sd2:
            want = torch.nn.functional.pixel_unshuffle(
                mean * ref.config.scaling_factor, 2
            )
        else:
            from diffusers.pipelines.flux2.pipeline_flux2 import Flux2Pipeline

            want = Flux2Pipeline._patchify_latents(mean)
            std = torch.sqrt(ref.bn.running_var + ref.config.batch_norm_eps)
            want = (want - ref.bn.running_mean.view(1, -1, 1, 1)) / std.view(
                1, -1, 1, 1
            )
        got = enc._latent(filled, 2, 3)  # 32 x 48 px = 2 x 3 patches at 16 px
    assert torch.allclose(got, want, atol=1e-4)
