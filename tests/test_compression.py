"""Pruning, quantisation and distillation — the claims this repo is built on.

Every defect the README records is pinned here, because each one is the kind
that leaves the code looking like it worked:

  * pruning removed its own mask, so the next optimiser step refilled the zeros
    and the sparsity quietly vanished;
  * `out_channels` was assigned directly, so the metadata said "narrower" while
    the convolution still computed at full width.

No pretrained weights are downloaded: the models here are small and local.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from model_compression import ModelCompressor


class Tiny(nn.Module):
    """Small, but with both layer types the compressor cares about."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 48, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))


@pytest.fixture
def compressor():
    return ModelCompressor(device="cpu")


# ── pruning ───────────────────────────────────────────────────────────────
def test_pruning_reaches_the_sparsity_it_was_asked_for(compressor):
    model = compressor.magnitude_pruning(Tiny(), sparsity=0.5)
    measured = ModelCompressor.measure_sparsity(model)
    assert measured == pytest.approx(0.5, abs=0.02), f"asked for 50%, got {measured:.1%}"


def test_a_model_starts_dense():
    assert ModelCompressor.measure_sparsity(Tiny()) < 0.01


def test_different_sparsities_are_honoured(compressor):
    for target in (0.2, 0.8):
        measured = ModelCompressor.measure_sparsity(compressor.magnitude_pruning(Tiny(), sparsity=target))
        assert measured == pytest.approx(target, abs=0.02)


def test_pruning_keeps_the_largest_weights(compressor):
    """Magnitude pruning removes the smallest weights, not arbitrary ones."""
    model = Tiny()
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[float(i) for i in range(48)] for _ in range(4)]))
    compressor.magnitude_pruning(model, sparsity=0.5)
    surviving = model.fc.weight.detach().abs()
    assert float(surviving[:, -1].abs().min()) > 0, "the largest column was pruned"
    assert float(surviving[:, 0].abs().max()) == 0, "the smallest column survived"


def test_sparsity_survives_an_optimiser_step(compressor):
    """The defect: prune.remove() straight away let training refill the zeros."""
    model = compressor.magnitude_pruning(Tiny(), sparsity=0.5)   # permanent=False by default
    optimiser = torch.optim.SGD(model.parameters(), lr=0.5)

    for _ in range(3):
        loss = model(torch.randn(2, 3, 16, 16)).sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    after = ModelCompressor.measure_sparsity(model)
    assert after == pytest.approx(0.5, abs=0.02), \
        f"sparsity fell to {after:.1%} after training — the mask is not being re-applied"


def test_permanent_pruning_bakes_the_zeros_in_and_drops_the_mask(compressor):
    model = compressor.magnitude_pruning(Tiny(), sparsity=0.5, permanent=True)
    assert not any(hasattr(m, "weight_orig") for m in model.modules()), \
        "permanent=True should strip the reparametrisation"
    assert ModelCompressor.measure_sparsity(model) == pytest.approx(0.5, abs=0.02)


def test_a_pruned_model_still_runs(compressor):
    model = compressor.magnitude_pruning(Tiny(), sparsity=0.6)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)


# ── the channel plan ──────────────────────────────────────────────────────
def test_the_channel_plan_changes_nothing(compressor):
    """It returns a proposal. Assigning out_channels would lie about the model."""
    model = Tiny()
    before = {name: (m.out_channels, tuple(m.weight.shape))
              for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}

    plan = compressor.channel_width_plan(model, factor=0.75)

    after = {name: (m.out_channels, tuple(m.weight.shape))
             for name, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    assert before == after, "channel_width_plan modified the model"
    assert plan, "no plan produced for a model with 64- and 48-channel convolutions"
    for name, proposed in plan.items():
        assert proposed < before[name][0], "a plan entry is not narrower"
        assert proposed >= 16


def test_the_plan_leaves_small_layers_alone(compressor):
    """Narrowing a 16-channel layer to 12 saves nothing and hurts accuracy."""
    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)

    assert compressor.channel_width_plan(Small(), factor=0.5) == {}


# ── distillation loss ─────────────────────────────────────────────────────
def test_matching_the_teacher_makes_the_soft_part_vanish(compressor):
    logits = torch.randn(8, 4)
    labels = logits.argmax(dim=1)

    same = compressor.knowledge_distillation_loss(logits, logits, labels, alpha=1.0)
    assert float(same) == pytest.approx(0.0, abs=1e-5), \
        "a student identical to the teacher has nothing to learn from it"


def test_disagreeing_with_the_teacher_costs_more(compressor):
    teacher = torch.tensor([[5.0, 0.0, 0.0, 0.0]] * 4)
    labels = torch.zeros(4, dtype=torch.long)
    agreeing = torch.tensor([[4.0, 0.2, 0.1, 0.1]] * 4)
    disagreeing = torch.tensor([[0.0, 0.0, 0.0, 5.0]] * 4)

    assert float(compressor.knowledge_distillation_loss(disagreeing, teacher, labels)) > \
           float(compressor.knowledge_distillation_loss(agreeing, teacher, labels))


def test_alpha_moves_the_weight_between_teacher_and_labels(compressor):
    """alpha=0 is plain cross-entropy against the labels; the teacher is ignored."""
    student = torch.randn(6, 4)
    teacher = torch.randn(6, 4)
    labels = torch.randint(0, 4, (6,))

    only_labels = compressor.knowledge_distillation_loss(student, teacher, labels, alpha=0.0)
    expected = nn.functional.cross_entropy(student, labels)
    assert float(only_labels) == pytest.approx(float(expected), abs=1e-5)


def test_the_loss_is_differentiable(compressor):
    student = torch.randn(4, 4, requires_grad=True)
    loss = compressor.knowledge_distillation_loss(student, torch.randn(4, 4),
                                                  torch.randint(0, 4, (4,)))
    loss.backward()
    assert student.grad is not None and torch.any(student.grad != 0)


# ── quantisation ──────────────────────────────────────────────────────────
def test_dynamic_quantisation_touches_linear_and_not_conv(compressor):
    """The README is explicit that this moves very little of a conv-heavy model."""
    quantised = compressor.quantize_model(Tiny())

    assert "quantized" in type(quantised.fc).__module__.lower() or \
           type(quantised.fc).__name__.lower().startswith("dynamicquantized"), \
           f"Linear was not quantised: {type(quantised.fc)}"
    assert isinstance(quantised.conv1, nn.Conv2d), \
        "dynamic quantisation does not support Conv2d; it must be left as it was"


def test_a_quantised_model_still_produces_the_right_shape(compressor):
    out = compressor.quantize_model(Tiny())(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 4)


def test_model_info_reports_real_numbers(compressor):
    info = compressor.get_model_info(Tiny())
    params = sum(p.numel() for p in Tiny().parameters())
    reported = info.get("total_parameters") or info.get("parameters")
    assert reported == params, f"reported {reported}, model has {params}"
