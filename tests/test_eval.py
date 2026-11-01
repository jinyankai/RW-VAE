"""Test for the E1 Accessibility-Gap evaluation."""
import eval_e1_ag as e1


def test_ag_identity_and_dominant_gap():
    rep = e1.demo(verbose=False)
    # exact telescoping identity: recon info loss + accessibility gap == total
    assert abs(rep["recon_info_loss"] + rep["accessibility_gap"] - rep["total"]) < 1e-9
    # manufactured case: small reconstruction info loss, large accessibility gap
    assert rep["accessibility_gap"] > 5 * abs(rep["recon_info_loss"])
