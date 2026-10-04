"""unit: masked_rel_loss == REL-only mean without a mask; masked dims carry 0 loss and 0 gradient; normalisation."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rel16_relonly_lossmask import masked_rel_loss

def test_no_mask_equals_relonly_mean():
    p, t = torch.randn(4, 16, 20), torch.randn(4, 16, 20)
    assert torch.equal(masked_rel_loss(p, t, None), torch.mean((p - t) ** 2))
    ones = torch.ones(4, 20, dtype=torch.bool)
    assert torch.allclose(masked_rel_loss(p, t, ones), torch.mean((p - t) ** 2), rtol=1e-6, atol=0)

def test_masked_dims_no_loss_no_grad_even_with_nan_targets():
    p = torch.randn(2, 16, 20, requires_grad=True); t = torch.randn(2, 16, 20)
    m = torch.zeros(2, 20, dtype=torch.bool); m[:, 10:19] = True
    t2 = t.clone(); t2[..., :10] = float("nan"); t2[..., 19] = 1e6
    l = masked_rel_loss(p, t2, m); l.backward()
    assert torch.isfinite(l) and p.grad[..., :10].abs().max() == 0 and p.grad[..., 19].abs().max() == 0 and p.grad[..., 10:19].abs().max() > 0
    assert torch.allclose(l, torch.mean((p[..., 10:19] - t[..., 10:19]) ** 2))

def test_mixed_batch_per_element_weight():
    p, t = torch.randn(2, 16, 20), torch.randn(2, 16, 20)
    m = torch.ones(2, 20, dtype=torch.bool); m[1, :10] = False; m[1, 19] = False
    se = (p - t) ** 2
    want = (se[0].sum() + se[1, :, 10:19].sum()) / (16 * 20 + 16 * 9)
    assert torch.allclose(masked_rel_loss(p, t, m), want)

if __name__ == "__main__":
    for f in (test_no_mask_equals_relonly_mean, test_masked_dims_no_loss_no_grad_even_with_nan_targets, test_mixed_batch_per_element_weight): f(); print("PASS", f.__name__)
