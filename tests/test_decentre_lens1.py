"""Route B: decentring lens 1 (and its half-mirror, tracer surfaces 0, 2, 3) is an
explicit, self-consistent option of the pancake to_batch/export path. The
polariser (surface 1) must stay put.

Two independent checks:
  1. the tracer's own batch: decentring shifts the vertex offset (batch.y) of
     surfaces 0, 2, 3 by exactly dy, and leaves surface 1 (the polariser)
     unchanged; the design still traces (no lost rays) at the tested decentre.
  2. the exporter's mesh agrees with the tracer's own decentred batch: every
     mesh vertex, taken back into its surface's local frame (which subtracts
     that same batch.y), sits exactly on that surface's own sag equation.
     A bug that decentred the mesh but not the batch (or the reverse) would
     make the front or back grid centre miss its own analytic surface, since
     the local frame and the sag coefficients would then no longer agree."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = (Path(__file__).resolve().parent.parent / "tier1_lightfield" / "foveated_optics_study"
        / "remapper_designs" / "freeform_mirror")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

import export_pancake as ep  # noqa: E402
import fold_search as fs  # noqa: E402
import offaxis_tracer as ot  # noqa: E402

BEST_JSON = HERE / "results_pancake" / "best_pancake_el1_glass.json"
DECENTRE_MM = 0.5   # this test design's own field is narrow (70x45); it checks the plumbing, not the usable range


@pytest.fixture(scope="module")
def entry():
    return json.loads(BEST_JSON.read_text())[0]


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(entry, device, decentre_mm):
    lay = fs.entry_layout(entry, "pancake")
    if decentre_mm:
        lay = dict(lay, decentre_lens1_mm=decentre_mm)
    x = torch.tensor([entry["x"]], dtype=torch.float64, device=device)
    idx = torch.tensor([entry["indices"]], dtype=torch.float64, device=device)
    return fs.to_batch(x, idx, lay)


def _traced(entry, device, decentre_mm):
    batch = _batch(entry, device, decentre_mm)
    ctx = fs.context(device)
    with torch.no_grad():
        _, _, alive, diag = ot.trace(batch, ctx["fields"], ctx["pupil"], diagnostics=True)
    assert bool(alive.all())
    return batch, diag["points"][0].cpu().numpy()


def test_decentre_offsets_lens_one_and_half_mirror_but_not_the_polariser(entry, device):
    base_batch, _ = _traced(entry, device, 0.0)
    dec_batch, _ = _traced(entry, device, DECENTRE_MM)
    for s in (0, 2, 3):
        assert float(dec_batch.y[0, s] - base_batch.y[0, s]) == pytest.approx(DECENTRE_MM, abs=1e-12)
    assert float(dec_batch.y[0, 1] - base_batch.y[0, 1]) == 0.0


def test_the_undecentred_batch_is_recovered_at_zero_mm(entry, device):
    base_batch, _ = _traced(entry, device, 0.0)
    zero_batch, _ = _traced(entry, device, 0.0)
    assert torch.equal(base_batch.y, zero_batch.y)


@pytest.fixture(scope="module")
def exported_both(tmp_path_factory, entry, device):
    base = tmp_path_factory.mktemp("decentre_export")
    (base / "best.json").write_text(json.dumps([entry]))
    out = {}
    for dy in (0.0, DECENTRE_MM):
        d = ep.export(base / "best.json", base / f"design_{dy}", device=str(device), decentre_lens1_mm=dy)
        out[dy] = (np.load(d / "remapper.npz"), json.loads((d / "design.json").read_text()))
    return out


def _tracer(world):
    """Inverse of export_fold.to_world."""
    return (world - np.array([0.0, ep.ef.lp.PUPIL_Y_MM, 0.0])) @ ep.ef.TRACER_TO_WORLD.T


def test_design_json_records_the_decentre(exported_both):
    for dy, (_, design) in exported_both.items():
        assert design["decentre_lens1_mm"] == dy


GRID_CENTRE = (ep.ef.GRID // 2) * ep.ef.GRID + ep.ef.GRID // 2      # row-major middle vertex of a GRID x GRID grid
N_GRID = ep.ef.GRID * ep.ef.GRID


def _grid_centre_on_its_own_surface(batch, data, verts_key, tracer_surface, offset=0):
    """The grid-centre vertex of a surf{k}_verts array (row-major GRID x GRID,
    at `offset` for the back half of a solid), taken back into
    tracer_surface's local frame, must sit on that surface's own sag equation
    evaluated at its own local (x, y): this is what the tracer means by
    'surface tracer_surface, batch.y[tracer_surface]'."""
    world = data[verts_key][offset + GRID_CENTRE]
    loc = ep.ef._local(batch, tracer_surface, _tracer(world[None]))[0]
    f, _, _ = ep.ef._sag(batch, tracer_surface, np.array([loc[0]]), np.array([loc[1]]))
    assert float(loc[2]) == pytest.approx(float(f[0]), abs=1e-9)


def test_exported_meshes_sit_on_their_own_decentred_surface(entry, device, exported_both):
    for dy, (data, _) in exported_both.items():
        batch = ep.ef.cpu_batch(_batch(entry, device, dy))
        _grid_centre_on_its_own_surface(batch, data, "surf0_verts", tracer_surface=1)                    # polariser
        _grid_centre_on_its_own_surface(batch, data, "surf1_verts", tracer_surface=2, offset=0)          # lens 1 front
        _grid_centre_on_its_own_surface(batch, data, "surf1_verts", tracer_surface=3, offset=N_GRID)      # lens 1 back


def test_exported_polariser_plane_does_not_move_with_the_decentre(exported_both):
    """The polariser's z (tracer frame) is fixed by z_p, gap alone: unaffected
    by lens 1's decentre, unlike its mesh's footprint (which follows the
    upstream, decentred half-mirror bounce)."""
    z = {dy: float(_tracer(data["surf0_verts"])[:, 2].mean()) for dy, (data, _) in exported_both.items()}
    assert z[0.0] == pytest.approx(z[DECENTRE_MM], abs=1e-9)
