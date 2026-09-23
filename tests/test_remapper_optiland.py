"""The optiland model of a coaxial remapper must trace exactly like the
independent torch tracer the coaxial design was built with, before optiland's
optimiser is trusted with it."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

STUDY = Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
sys.path.insert(0, str(STUDY / "scripts"))
sys.path.insert(0, str(STUDY / "remapper_designs" / "coaxial_dls"))
sys.path.insert(0, str(STUDY / "remapper_designs" / "coaxial" / "src"))

import coaxial_optiland as co  # noqa: E402

# The agent's design_opt sets torch's global default dtype to float64 when it is
# imported; restore it so this module does not change every later test.
_default_dtype = torch.get_default_dtype()
import design_opt as agent  # noqa: E402
torch.set_default_dtype(_default_dtype)

FIELDS_DEG = [0.0, 5.0, 15.0, 25.0, 35.0]


@pytest.fixture(scope="module")
def seed():
    params = agent.Params(agent.N_ELEMENTS)
    params.load_state_dict(torch.load(STUDY / "remapper_designs" / "coaxial" / "checkpoint.pt"))
    return params, co.prescription_from_agent(params)


def _agent_hit(params, theta_deg, pupil_xz):
    theta = torch.tensor([math.radians(theta_deg)], dtype=torch.float64)
    pupil = torch.tensor([pupil_xz], dtype=torch.float64)
    with torch.no_grad():
        hit, _, alive, *_ = agent.trace_all(params, theta, pupil, 60.0)
    assert bool(alive.all())
    return hit[0, 0].numpy()


def test_normalised_field_is_the_intended_angle(seed):
    _, rx = seed
    lens = co.build(rx, FIELDS_DEG)
    for theta in FIELDS_DEG[1:]:
        lens.trace_generic(0.0, theta / max(FIELDS_DEG), 0.0, 0.0, co.WAVELENGTH_UM)
        m, n = float(lens.surfaces.M[1, 0]), float(lens.surfaces.N[1, 0])
        assert math.degrees(math.atan2(m, n)) == pytest.approx(theta, abs=1e-9)


@pytest.mark.parametrize("theta", FIELDS_DEG)
@pytest.mark.parametrize("py", [0.0, 1.0, -1.0])
def test_optiland_lands_where_the_agent_tracer_lands(seed, theta, py):
    params, rx = seed
    lens = co.build(rx, FIELDS_DEG)
    lens.trace_generic(0.0, theta / max(FIELDS_DEG), 0.0, py, co.WAVELENGTH_UM)
    y_opt = float(lens.surfaces.y[-1, 0])
    z_agent = _agent_hit(params, theta, (0.0, py * co.PUPIL_RADIUS_MM))[2]
    assert y_opt == pytest.approx(z_agent, abs=1e-6)


def test_parameter_vector_round_trips_the_prescription(seed):
    import fast_merit as fm
    _, rx = seed
    indices = [e["index"] for e in rx["elements"]]
    back = fm.to_prescription(fm.to_vector(rx), indices)
    for a, b in zip(rx["elements"], back["elements"]):
        for side in ("front", "back"):
            assert b[side]["radius_mm"] == pytest.approx(a[side]["radius_mm"], rel=1e-12)
            assert b[side]["conic"] == pytest.approx(a[side]["conic"], abs=1e-15)
            assert b[side]["coefficients"][1:] == pytest.approx(a[side]["coefficients"][1:], abs=1e-18)
        assert (b["thickness_mm"], b["gap_after_mm"]) == pytest.approx((a["thickness_mm"], a["gap_after_mm"]))


def test_one_trace_residuals_match_optiland_operands(seed):
    """The chief-ray residual of the vectorised trace equals the real_y_intercept
    operand, and its landing residuals reproduce optiland's RMS spot radius
    when the target is the spot centroid instead of r(theta)."""
    import fast_merit as fm
    import merit
    _, rx = seed
    indices = [e["index"] for e in rx["elements"]]
    x = fm.to_vector(rx)
    fields = [0.0, 10.0, 30.0]
    res = fm.residuals(x, indices, fields_deg=fields)
    n_pupil = len(fm.pupil_samples())
    chief = res[len(fields) * n_pupil:len(fields) * n_pupil + len(fields)] / fm.W_CHIEF
    lens = co.build(rx, [0.0, merit.MAX_FIELD_DEG])
    rows = merit.measure(lens, 1.0, fields=fields)
    # batched vs single-ray traces differ at optiland's Newton tolerance (~0.2 nm)
    assert chief == pytest.approx([r["height_err_mm"] for r in rows], abs=1e-6)
    lens, pupil = fm.trace(rx, fields)
    img = merit.image_surface(lens)
    Y = np.asarray(lens.surfaces.y[img]).reshape(len(fields), len(pupil))
    X = np.asarray(lens.surfaces.x[img]).reshape(len(fields), len(pupil))
    ours = np.sqrt(np.mean((X - X.mean(1, keepdims=True))**2 + (Y - Y.mean(1, keepdims=True))**2, axis=1))
    assert ours == pytest.approx([r["spot_rms_mm"] for r in rows], rel=0.15)


def test_image_plane_is_where_the_agent_put_the_mla(seed):
    params, rx = seed
    with torch.no_grad():
        _, image_y = params.build(60.0)
    assert co.PUPIL_Y_MM + rx["eye_relief_mm"] + sum(
        e["thickness_mm"] + e["gap_after_mm"] for e in rx["elements"]) == pytest.approx(float(image_y), abs=1e-9)
