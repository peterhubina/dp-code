"""Tests for the film_attention and coattn fusion modes.

Run from project/CLAM:   python tests/test_fusion_modes.py

No pytest in this environment, so this is a plain script: it prints one line per check and
exits non-zero if any fails. Every test is written so that breaking the mechanism makes it
fail; this was confirmed by deliberately breaking the mechanism four ways and
checking each was caught (recorded in docs/implementation-research/novel-fusion-design.md 7.4).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

CLAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLAM_ROOT))

from models.model_clam import CLAM_MB  # noqa: E402
from models.model_multimodal import FUSION_MODES, CLAMRNAFusion  # noqa: E402
from utils.tabular_groups import MAX_TOKENS, build_tabular_groups  # noqa: E402

EMBED_DIM = 1536
N_PATCHES = 137
N_CLASSES = 2
RNA_DIM = 20530
CLIN_DIM = 24

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def build(fusion_mode, tabular_dim, *, film_rank=32, modality_dropout=0.0, groups=None, seed=0):
    torch.manual_seed(seed)
    return CLAMRNAFusion(
        wsi_model_type="clam_mb",
        size_arg="big",
        embed_dim=EMBED_DIM,
        n_classes=N_CLASSES,
        dropout=0.0,
        tabular_input_dim=tabular_dim,
        tabular_hidden_dim=256,
        tabular_num_layers=2,
        fusion_hidden_dim=32,
        fusion_mode=fusion_mode,
        film_rank=film_rank,
        modality_dropout=modality_dropout,
        tabular_group_indices=groups,
    )


def sample(tabular_dim, seed=0):
    torch.manual_seed(seed + 1000)
    return torch.randn(1, N_PATCHES, EMBED_DIM), torch.randn(1, tabular_dim)


# --------------------------------------------------------------------------------------
# 1. Output shapes for both a 20530-dim and a 24-dim tabular input
# --------------------------------------------------------------------------------------
def test_output_shapes():
    clin_groups = [list(range(0, 8)), list(range(8, 16)), list(range(16, CLIN_DIM))]
    rna_groups = [list(range(0, 10000)), list(range(10000, RNA_DIM))]
    cases = [
        ("film_attention", RNA_DIM, None),
        ("film_attention", CLIN_DIM, None),
        ("coattn", RNA_DIM, rna_groups),
        ("coattn", CLIN_DIM, clin_groups),
    ]
    for mode, dim, groups in cases:
        model = build(mode, dim, groups=groups).eval()
        wsi, tab = sample(dim)
        logits, y_prob, y_hat, attention, results = model((wsi, tab))
        ok = (
            logits.shape == (1, N_CLASSES)
            and y_prob.shape == (1, N_CLASSES)
            and y_hat.shape == (1, 1)
            and torch.isfinite(logits).all()
            and abs(float(y_prob.sum()) - 1.0) < 1e-5
        )
        check(f"shapes: {mode} with {dim}-dim tabular", ok,
              f"logits {tuple(logits.shape)}, attention {tuple(attention.shape)}, "
              f"metrics {sorted(results)}")


# --------------------------------------------------------------------------------------
# 2. FiLM at initialisation reproduces the WSI-alone logits EXACTLY
#    (this is the do-no-harm property the design claims)
# --------------------------------------------------------------------------------------
def test_film_identity_equals_wsi_alone():
    for dim in (RNA_DIM, CLIN_DIM):
        model = build("film_attention", dim).eval()
        wsi, tab = sample(dim)
        fusion_logits, _, _, _, _ = model((wsi, tab))
        with torch.no_grad():
            wsi_logits, _, _, _, _ = model.wsi(wsi, return_features=True)
        diff = float((fusion_logits - wsi_logits).abs().max())
        check(f"film at init == WSI-alone logits ({dim}-dim)", diff < 1e-6, f"max|diff| = {diff:.3e}")


def test_film_rank_zero_is_additive_logit_fusion():
    """film_rank=0 must disable attention conditioning entirely (the Design-A ablation)."""
    model = build("film_attention", RNA_DIM, film_rank=0).eval()
    has_film = any(hasattr(model, attr) for attr in ("film_bottleneck", "film_gamma", "film_beta"))
    check("film_rank=0 creates no FiLM parameters", not has_film)
    wsi, tab = sample(RNA_DIM)
    logits, _, _, _, results = model((wsi, tab))
    with torch.no_grad():
        wsi_logits, _, _, _, _ = model.wsi(wsi, return_features=True)
    diff = float((logits - wsi_logits).abs().max())
    check("film_rank=0 at init == WSI-alone logits", diff < 1e-6, f"max|diff| = {diff:.3e}")
    check("film_rank=0 logs no gamma metric", "fusion_film_gamma_dev" not in results)


def test_film_actually_changes_attention():
    """A non-identity FiLM must re-rank patches, not merely rescale the attention."""
    model = build("film_attention", CLIN_DIM).eval()
    wsi, tab = sample(CLIN_DIM)
    with torch.no_grad():
        _, _, _, baseline_attention, _ = model((wsi, tab))
        # Perturb the FiLM generators away from identity.
        torch.nn.init.normal_(model.film_gamma.weight, std=0.5)
        torch.nn.init.normal_(model.film_beta.weight, std=0.5)
        _, _, _, modulated_attention, _ = model((wsi, tab))
    base_rank = torch.argsort(baseline_attention[0])
    mod_rank = torch.argsort(modulated_attention[0])
    changed = int((base_rank != mod_rank).sum())
    check("non-identity FiLM re-ranks patches", changed > N_PATCHES // 4,
          f"{changed}/{N_PATCHES} rank positions moved")


# --------------------------------------------------------------------------------------
# 3. Gradients reach the new parameters
# --------------------------------------------------------------------------------------
def test_pooling_uses_unmodulated_embeddings():
    """The core design property: FiLM re-ranks patches but must NOT distort the pooled
    representation. Pooling the modulated embeddings instead would silently break this and
    every other test would still pass, so it is checked explicitly against a manual
    recomputation with a deliberately non-identity FiLM."""
    model = build("film_attention", CLIN_DIM).eval()
    wsi, tab = sample(CLIN_DIM)
    torch.nn.init.normal_(model.film_gamma.weight, std=0.3)
    torch.nn.init.normal_(model.film_beta.weight, std=0.3)

    with torch.no_grad():
        _, _, _, attention_raw, results = model((wsi, tab), return_features=True)
        patches = wsi.squeeze(0)
        h = model.wsi.attention_net[:3](patches)
        bottleneck = model.film_bottleneck(model.tabular_encoder(tab))
        gamma = 1.0 + model.film_gamma(bottleneck)
        beta = model.film_beta(bottleneck)
        modulated = gamma * h + beta

        attention = torch.softmax(attention_raw, dim=1)
        pooled_from_original = torch.mm(attention, h).mean(dim=0, keepdim=True)
        pooled_from_modulated = torch.mm(attention, modulated).mean(dim=0, keepdim=True)

    reported = results["fusion_features"]
    d_original = float((reported - pooled_from_original).abs().max())
    d_modulated = float((reported - pooled_from_modulated).abs().max())
    check("pooled feature is built from the UNMODULATED patch embeddings", d_original < 1e-5,
          f"|pooled - original| = {d_original:.3e}")
    check("pooled feature is NOT the modulated one (test discriminates)", d_modulated > 1e-4,
          f"|pooled - modulated| = {d_modulated:.3e}")


def test_gradient_flow():
    specs = [
        ("film_attention", RNA_DIM, None,
         ["film_gamma.weight", "film_beta.weight", "tabular_head.weight"]),
        ("coattn", CLIN_DIM, [list(range(0, 12)), list(range(12, CLIN_DIM))],
         ["patch_projection.weight", "tabular_token_encoders.0.weight", "image_head.weight"]),
    ]
    for mode, dim, groups, expected in specs:
        model = build(mode, dim, groups=groups).train()
        wsi, tab = sample(dim)
        logits, _, _, _, _ = model((wsi, tab))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        named = dict(model.named_parameters())
        for pname in expected:
            grad = named[pname].grad
            ok = grad is not None and torch.isfinite(grad).all() and float(grad.abs().sum()) > 0
            check(f"gradient reaches {mode}.{pname}", ok,
                  "no grad" if grad is None else f"|grad|sum = {float(grad.abs().sum()):.3e}")


def test_film_bottleneck_activates_after_first_step():
    """film_gamma/beta are zero-initialised for the identity property, which necessarily
    leaves film_bottleneck with zero gradient on step 0 (chain rule through a zero matrix).
    It must become active once the output layers move off zero -- the ControlNet
    zero-convolution pattern. This test pins that behaviour down in both directions."""
    model = build("film_attention", CLIN_DIM)
    model.freeze_wsi_branch()
    model.train()
    wsi, tab = sample(CLIN_DIM)
    target = torch.tensor([1])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    losses, bottleneck_grads = [], []
    for _ in range(3):
        logits, _, _, _, _ = model((wsi, tab))
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        grad = model.film_bottleneck.weight.grad
        bottleneck_grads.append(0.0 if grad is None else float(grad.abs().sum()))
        losses.append(float(loss))
        optimizer.step()

    check("film_bottleneck is dormant at init (expected: zero-init outputs)",
          bottleneck_grads[0] == 0.0, f"step-0 |grad| = {bottleneck_grads[0]:.3e}")
    check("film_bottleneck receives gradient from step 1 onward",
          bottleneck_grads[1] > 0, f"step-1 |grad| = {bottleneck_grads[1]:.3e}")
    check("film parameters move away from identity during training",
          float(model.film_gamma.weight.abs().sum()) > 0)
    check("loss decreases over three steps", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")


# --------------------------------------------------------------------------------------
# 4. --freeze_wsi_branch genuinely freezes the WSI branch
# --------------------------------------------------------------------------------------
def test_freeze_wsi_branch():
    for mode, dim, groups in [("film_attention", RNA_DIM, None),
                              ("coattn", CLIN_DIM, [list(range(0, 12)), list(range(12, CLIN_DIM))])]:
        model = build(mode, dim, groups=groups)
        model.freeze_wsi_branch()
        all_frozen = all(not p.requires_grad for p in model.wsi.parameters())
        check(f"{mode}: all WSI params requires_grad=False", all_frozen)

        before = {n: p.detach().clone() for n, p in model.wsi.named_parameters()}
        trainable = [p for p in model.parameters() if p.requires_grad]
        check(f"{mode}: some non-WSI params remain trainable", len(trainable) > 0,
              f"{sum(p.numel() for p in trainable)} trainable params")
        optimizer = torch.optim.Adam(trainable, lr=1e-2)

        model.train()
        wsi, tab = sample(dim)
        logits, _, _, _, _ = model((wsi, tab))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        optimizer.step()

        moved = [n for n, p in model.wsi.named_parameters()
                 if not torch.equal(p.detach(), before[n])]
        check(f"{mode}: WSI params unchanged after an optimiser step", not moved,
              f"moved: {moved[:3]}" if moved else "")
        grads = [n for n, p in model.wsi.named_parameters() if p.grad is not None]
        check(f"{mode}: WSI params received no gradients", not grads, f"got grads: {grads[:3]}")

        # The training step must actually have changed the trainable parameters, otherwise
        # "WSI unchanged" would pass vacuously.
        head_changed = float(model.tabular_head.weight.abs().sum()) > 0
        check(f"{mode}: trainable head did move (test is not vacuous)", head_changed)


# --------------------------------------------------------------------------------------
# 5. Modality dropout / missing-modality inference
# --------------------------------------------------------------------------------------
def test_modality_absent_inference():
    model = build("film_attention", RNA_DIM, modality_dropout=0.25).eval()
    wsi, tab = sample(RNA_DIM)
    check("modality_dropout creates an absent embedding", hasattr(model, "tabular_absent_embedding"))

    model.force_tabular_absent = True
    logits_absent, prob_absent, _, _, _ = model((wsi, tab))
    with torch.no_grad():
        wsi_logits, _, _, _, _ = model.wsi(wsi, return_features=True)
    ok = torch.isfinite(logits_absent).all() and abs(float(prob_absent.sum()) - 1.0) < 1e-5
    check("absent-modality inference produces a valid distribution", ok)
    diff = float((logits_absent - wsi_logits).abs().max())
    check("absent-modality inference falls back to WSI-alone at init", diff < 1e-6,
          f"max|diff| = {diff:.3e}")

    # The absent path must ignore the tabular vector entirely.
    _, tab_other = sample(RNA_DIM, seed=99)
    logits_other, _, _, _, _ = model((wsi, tab_other))
    check("absent-modality output is invariant to the tabular input",
          float((logits_absent - logits_other).abs().max()) < 1e-6)

    model.force_tabular_absent = False
    torch.nn.init.normal_(model.film_gamma.weight, std=0.1)
    torch.nn.init.normal_(model.tabular_head.weight, std=0.1)
    logits_present, _, _, _, _ = model((wsi, tab))
    check("present-modality output differs from absent once trained away from init",
          float((logits_present - logits_absent).abs().max()) > 1e-6)


def test_modality_dropout_only_active_in_training():
    model = build("film_attention", CLIN_DIM, modality_dropout=0.9)
    wsi, tab = sample(CLIN_DIM)
    torch.nn.init.normal_(model.tabular_head.weight, std=0.5)
    model.eval()
    with torch.no_grad():
        runs = {float(model((wsi, tab))[0][0, 0]) for _ in range(12)}
    check("eval mode is deterministic despite modality_dropout=0.9", len(runs) == 1,
          f"{len(runs)} distinct outputs")


# --------------------------------------------------------------------------------------
# 6. The four pre-existing modes are byte-for-byte unchanged (regression vs git HEAD)
# --------------------------------------------------------------------------------------
def test_existing_modes_unchanged():
    # Pinned to the last commit that touched model_multimodal.py BEFORE the new modes were
    # added. Deliberately not HEAD: once this change is committed, comparing against HEAD
    # would compare the file with itself and pass vacuously.
    baseline_rev = "60a9639133dfabd335ede43feeef55cb5db3da3a"
    original_path = Path(tempfile.gettempdir()) / f"model_multimodal_{baseline_rev[:8]}.py"
    if not original_path.is_file():
        try:
            source = subprocess.check_output(
                ["git", "show", f"{baseline_rev}:project/CLAM/models/model_multimodal.py"],
                cwd=CLAM_ROOT, text=True, stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            check("regression vs pre-change baseline", False,
                  f"could not retrieve {baseline_rev[:8]}: {exc}")
            return
        original_path.write_text(source)

    spec = importlib.util.spec_from_file_location("model_multimodal_orig", original_path)
    original = importlib.util.module_from_spec(spec)
    sys.modules["model_multimodal_orig"] = original
    spec.loader.exec_module(original)

    for mode in ("concat", "gated", "cross_attention", "residual"):
        wsi, tab = sample(CLIN_DIM)
        kwargs = dict(
            wsi_model_type="clam_mb", size_arg="big", embed_dim=EMBED_DIM, n_classes=N_CLASSES,
            dropout=0.0, tabular_input_dim=CLIN_DIM, tabular_hidden_dim=256,
            tabular_num_layers=2, fusion_hidden_dim=32, fusion_mode=mode,
        )
        if mode == "residual":
            # residual builds an RNA_MLP branch instead of the shared tabular encoder.
            kwargs.update(rna_hidden_dims=(64, 32), rna_dropout=0.4, residual_scale=0.2)
        torch.manual_seed(0)
        old = original.CLAMRNAFusion(**kwargs).eval()
        torch.manual_seed(0)
        new = CLAMRNAFusion(**kwargs).eval()
        with torch.no_grad():
            old_logits = old((wsi, tab))[0]
            new_logits = new((wsi, tab))[0]
        diff = float((old_logits - new_logits).abs().max())
        check(f"existing mode '{mode}' unchanged vs {baseline_rev[:8]}", diff == 0.0, f"max|diff| = {diff:.3e}")

    check("FUSION_MODES still contains all four original modes",
          {"concat", "gated", "residual", "cross_attention"}.issubset(set(FUSION_MODES)))


# --------------------------------------------------------------------------------------
# 7. Token grouping
# --------------------------------------------------------------------------------------
def test_tabular_groups():
    clin = ["age", "ajcc_stage_I", "ajcc_stage_II", "pathologic_t_T1", "pathologic_t_T2",
            "histological_type_ductal", "histological_type_unknown"]
    names, indices = build_tabular_groups(clin, "prefix")
    ok = (names == ["age", "ajcc_stage", "histological_type", "pathologic_t"]
          and sum(len(i) for i in indices) == len(clin))
    check("prefix grouping recovers clinicopath blocks", ok, f"{names}")

    signatures = CLAM_ROOT.parent / "MCAT" / "dataset_csv" / "signatures.csv"
    if signatures.is_file():
        genes = ["ESR1", "GATA3", "NOT_A_REAL_GENE_XYZ"]
        names, indices = build_tabular_groups(genes, str(signatures))
        covered = sorted({i for group in indices for i in group})
        check("signature grouping covers every feature exactly once or more",
              covered == [0, 1, 2] and "unassigned" in names,
              f"groups={names}")
        esr1_groups = [n for n, idx in zip(names, indices) if 0 in idx]
        check("ESR1 lands in a signature group (not 'unassigned')",
              esr1_groups and esr1_groups != ["unassigned"], f"ESR1 -> {esr1_groups}")
    else:
        check("signature CSV present", False, f"missing {signatures}")

    # A signature CSV that matches nothing must FAIL LOUDLY, not silently collapse into a
    # single 'unassigned' token -- that would train fine and look like a valid baseline
    # while testing nothing.
    bogus = Path(tempfile.gettempdir()) / "bogus_signatures.csv"
    bogus.write_text("GroupA,GroupB\nNOPE1,NOPE3\nNOPE2,NOPE4\n")
    try:
        build_tabular_groups(["ESR1", "GATA3", "TP53"], str(bogus))
        check("non-matching signature CSV raises", False, "it returned groups instead")
    except ValueError:
        check("non-matching signature CSV raises", True)

    # 'prefix' on bare gene symbols would make one token per gene; that must be refused.
    try:
        build_tabular_groups([f"GENE{i}" for i in range(MAX_TOKENS + 5)], "prefix")
        check("'prefix' refuses to produce a runaway token count", False, "it returned groups")
    except ValueError:
        check("'prefix' refuses to produce a runaway token count", True)


def main():
    print("=" * 78)
    for fn in (test_output_shapes, test_film_identity_equals_wsi_alone,
               test_film_rank_zero_is_additive_logit_fusion, test_film_actually_changes_attention,
               test_pooling_uses_unmodulated_embeddings,
               test_gradient_flow, test_film_bottleneck_activates_after_first_step,
               test_freeze_wsi_branch, test_modality_absent_inference,
               test_modality_dropout_only_active_in_training, test_existing_modes_unchanged,
               test_tabular_groups):
        print(f"\n--- {fn.__name__} ---")
        fn()
    print("=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
