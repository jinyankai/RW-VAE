"""rwvae.tokenizer_v3: shapes, fusion modes, gate initialisation, readout modes, KL / rate lock, edges,
decoder weight loading and checkpoint round trip, on tiny CPU models (32 px images, 8x8 latents)."""
import math

import pytest
import torch

from rwvae import tokenizer_v3 as tv3

pytest.importorskip("einops")

GRID, IMG, D, L = 8, 32, 12, 2


def tiny_cfg(**kw):
    base = dict(dino_layers=(0, 1), dino_dim=D, feat_dim=16, groups=4, latent_dim=8, rgb_channels=(8, 16),
                gate_hidden=8, sem_depth=1, sem_width=16, sem_heads=2, det_width=16)
    base.update(kw)
    return tv3.V3Config(**base)


def tiny_tok(**kw):
    torch.manual_seed(0)
    return tv3.TokenizerV3(tiny_cfg(**kw), tv3.build_decoder_ae(8, (1, 2, 2)), grid=GRID)


def batch(B=3):
    g = torch.Generator().manual_seed(1)
    images = torch.rand(B, 3, IMG, IMG, generator=g) * 2 - 1
    dino = torch.randn(B, L, D, GRID, GRID, generator=g)
    return images, dino


@pytest.mark.parametrize("fusion", tv3.FUSIONS)
def test_encode_decode_shapes(fusion):
    tok = tiny_tok(fusion=fusion)
    images, dino = batch()
    out = tok.encode(images, dino)
    assert out["mu"].shape == out["logvar"].shape == (3, 8, GRID, GRID)
    assert float(out["logvar"].min()) >= -12 and float(out["logvar"].max()) <= 4
    if fusion == "dino_only":
        assert out["alpha"] is None and not hasattr(tok, "rgb")
    else:
        assert out["alpha"].shape == (3, 4, GRID, GRID)
    assert hasattr(tok, "gate") == (fusion == "adaptive")
    x = tok.decode(tok.sample(out["mu"], out["logvar"]))
    assert x.shape == (3, 3, IMG, IMG)


def test_adaptive_gate_starts_at_half_and_fixed_is_half():
    images, dino = batch()
    a = tiny_tok(fusion="adaptive").encode(images, dino)["alpha"]
    f = tiny_tok(fusion="fixed").encode(images, dino)["alpha"]
    assert torch.allclose(a, torch.full_like(a, 0.5)) and torch.allclose(f, torch.full_like(f, 0.5))


def test_freeze_gate_pins_half_with_zero_gate_gradient():
    tok = tiny_tok(fusion="adaptive")
    images, dino = batch()
    with torch.no_grad():
        tok.gate[-1].bias.fill_(5.0)                     # an unfrozen gate would give alpha ~ 0.993
    out = tok(images, dino, torch.randn(3, D, GRID, GRID), freeze_gate=True)
    assert torch.equal(out["alpha"], torch.full_like(out["alpha"], 0.5))
    (out["mu"].sum() + out["sem"]).backward()
    assert tok.gate[-1].weight.grad is not None and float(tok.gate[-1].weight.grad.abs().max()) == 0.0
    assert float(tok.dino_proj.weight.grad.abs().max()) > 0
    assert float(tok.encode(images, dino)["alpha"].min()) > 0.99


def test_gate_mixes_groups_independently():
    """Group g of the posterior head only reads feature group g, mixed by alpha[:, g]."""
    tok = tiny_tok(fusion="adaptive")
    images, dino = batch(1)
    with torch.no_grad():
        tok.gate[-1].bias.copy_(torch.tensor([20.0, -20.0, 0.0, 0.0]))   # group 0 -> DINO, group 1 -> RGB
    out = tok.encode(images, dino)
    a = out["alpha"]
    assert float(a[:, 0].min()) > 0.999 and float(a[:, 1].max()) < 1e-3
    h_expected = out["d"][:, :4] * a[:, :1] + out["r"][:, :4] * (1 - a[:, :1])
    params = tok.posterior(torch.cat([h_expected, torch.zeros(1, 12, GRID, GRID)], 1))
    # group 0 owns posterior channels [mu_0, mu_1, logvar_0, logvar_1]; zeroing groups 1..3 does not change it
    assert torch.allclose(params[:, :2], out["mu"][:, :2], atol=1e-5)
    assert torch.allclose(params[:, 2:4].clamp(-12, 4), out["logvar"][:, :2], atol=1e-5)


def test_rgb_branch_affects_latent_only_when_used():
    images, dino = batch()
    other = torch.flip(images, dims=[-1])
    for fusion, changes in (("dino_only", False), ("fixed", True)):
        tok = tiny_tok(fusion=fusion)
        same = torch.allclose(tok.encode(images, dino)["mu"], tok.encode(other, dino)["mu"])
        assert same != changes


def test_kl_and_rate():
    mu, logvar = torch.zeros(2, 8, 4, 4), torch.zeros(2, 8, 4, 4)
    assert float(tv3.rate_nats_per_token(mu, logvar)) == 0.0
    mu = torch.ones(2, 8, 4, 4)
    assert math.isclose(float(tv3.rate_nats_per_token(mu, logvar)), 8 * 0.5, rel_tol=1e-6)
    logvar = torch.full_like(mu, math.log(0.25))
    expect = 0.5 * (1 + 0.25 - 1 - math.log(0.25))
    assert math.isclose(float(tv3.kl_elementwise(mu, logvar)[0, 0, 0, 0]), expect, rel_tol=1e-6)


def test_rate_lagrangian_moves_lambda_towards_target():
    lag = tv3.RateLagrangian(20.0, every=2, lr=0.1)
    loss, delta = lag.loss(torch.tensor(30.0))
    assert math.isclose(delta, 0.5) and math.isclose(float(loss), 5 * 0.25)
    assert not lag.observe(delta) and lag.observe(delta)
    assert math.isclose(lag.lam, 0.05)
    lag2 = tv3.RateLagrangian(20.0)
    lag2.load_state_dict(lag.state_dict())
    assert lag2.lam == lag.lam


def test_readout_modes_and_gradients():
    images, dino = batch()
    target = torch.randn(3, D, GRID, GRID)
    for mode in ("split", "shared"):
        tok = tiny_tok(readout=mode)
        out = tok.encode(images, dino)
        z = tok.sample(out["mu"], out["logvar"])
        tok.update_moments(z)
        losses = tok.readout_losses(z, images, target, generator=torch.Generator().manual_seed(0))
        (losses["sem"] + losses["det"]).backward()
        assert tok.dino_proj.weight.grad is not None and float(tok.dino_proj.weight.grad.abs().sum()) > 0
        assert 0.0 <= float(losses["sem"]) <= 2.0
    t_sem, t_det = tiny_tok(readout="split").readout_times(4, "cpu")
    assert torch.all(t_sem == 0.35) and torch.all(t_det == 0.80)
    t1, t2 = tiny_tok(readout="shared").readout_times(4, "cpu", torch.Generator().manual_seed(0))
    assert torch.equal(t1, t2) and len(set(t1.tolist())) == 4
    tok = tiny_tok(readout="off")
    assert not hasattr(tok, "sem_head")


def test_vf_loss_margins_projection_and_forward():
    tok = tiny_tok(readout="off", vf=True)
    assert tok.vf_proj.bias is None and tok.vf_proj.weight.shape == (D, 8, 1, 1)
    assert not hasattr(tiny_tok(), "vf_proj")
    images, dino = batch()
    z = torch.randn(3, 8, GRID, GRID)
    target = torch.randn(3, D, GRID, GRID)
    # z' == target: both terms vanish (cos = 1, identical cosine matrices)
    with torch.no_grad():
        assert float(tok.vf_loss(z, tok.vf_proj(z))) < 1e-6
    # zero margins: the loss is the plain |cosine-matrix diff| + (1 - cos), and margins only lower it
    strict = tiny_tok(readout="off", vf=True, vf_cos_margin=0.0, vf_distmat_margin=0.0)
    strict.vf_proj.load_state_dict(tok.vf_proj.state_dict())
    zn = torch.nn.functional.normalize(tok.vf_proj(z).flatten(2), dim=1)
    fn = torch.nn.functional.normalize(target.flatten(2), dim=1)
    ref = ((zn.transpose(1, 2) @ zn - fn.transpose(1, 2) @ fn).abs().mean() + (1 - (zn * fn).sum(1)).mean())
    assert torch.allclose(strict.vf_loss(z, target), ref, atol=1e-5)
    assert float(tok.vf_loss(z, target)) <= float(ref) + 1e-6
    out = tok(images, dino, target)
    assert "vf" in out and "sem" not in out
    out["vf"].backward()
    assert float(tok.dino_proj.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        tok(images, dino)
    tiny_tok(readout="off")(images, dino)          # neither readouts nor VF: no target needed


def test_update_moments_first_call_copies_then_ema():
    tok = tiny_tok()
    z = torch.randn(4, 8, GRID, GRID) * 3 + 1
    tok.update_moments(z)
    assert torch.allclose(tok.z_mean, z.mean((0, 2, 3)), atol=1e-5)
    u = tok.standardize(z)
    assert torch.allclose(u.mean((0, 2, 3)), torch.zeros(8), atol=1e-4)
    tok.update_moments(torch.zeros_like(z))
    assert torch.allclose(tok.z_mean, 0.99 * z.mean((0, 2, 3)), atol=1e-5)


def test_edge_map():
    flat = torch.zeros(1, 3, 32, 32)
    assert float(tv3.edge_map(flat, 16).max()) < 1e-5
    step = torch.full((1, 3, 32, 32), -1.0)
    step[..., 16:] = 1.0
    e = tv3.edge_map(step, 16)[0, 0]
    assert e.shape == (16, 16) and float(e[:, 7:9].min()) > 0.1 and float(e[:, :4].max()) < 1e-5


def test_parameter_groups_partition():
    tok = tiny_tok()
    enc, dec = tok.encoder_parameters(), tok.decoder_parameters()
    assert len(enc) + len(dec) == len(list(tok.parameters()))
    assert not {id(p) for p in enc} & {id(p) for p in dec}


def test_decoder_weights_and_checkpoint_roundtrip(tmp_path):
    ae = tv3.build_decoder_ae(8, (1, 2, 2))
    tok = tiny_tok()
    info = tv3.load_decoder_weights(tok, ae.state_dict())
    assert info["num_tensors"] > 0
    assert torch.equal(tok.post_quant_conv.weight, ae.post_quant_conv.weight)
    with pytest.raises(ValueError):
        tv3.load_decoder_weights(tok, {})
    path = tmp_path / "tok.pt"
    tv3.save_tokenizer(path, tok, decoder_arch={"embed_dim": 8, "ch_mult": [1, 2, 2]}, extra={"step": 7})
    tok2, ckpt = tv3.load_tokenizer(path)
    assert ckpt["step"] == 7 and tok2.cfg == tok.cfg
    images, dino = batch()
    assert torch.allclose(tok.eval().encode(images, dino)["mu"], tok2.encode(images, dino)["mu"])


def test_config_validation():
    with pytest.raises(ValueError):
        tiny_cfg(fusion="concat")
    with pytest.raises(ValueError):
        tiny_cfg(readout="sometimes")
    with pytest.raises(ValueError):
        tiny_cfg(rgb_channels=(8, 12))
    with pytest.raises(ValueError):
        tv3.V3Config.from_dict({"nope": 1})
    cfg = tv3.V3Config.from_dict({"dino_layers": [16, 23], "fusion": "fixed"})
    assert cfg.dino_layers == (16, 23) and cfg.target_layer == 23
    assert tv3.V3Config.from_dict(cfg.to_dict()) == cfg
