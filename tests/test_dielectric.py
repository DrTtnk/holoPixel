"""
The glass sphere in the Cornell scene.

A sign error in Snell's law or in the Fresnel terms produces a picture that
still looks like glass, so these are checked numerically against closed forms
and symbolically with sympy rather than judged by eye. The eye check is worth
doing too and it is: a solid sphere is a lens, so the walls appear swapped
inside it (plots/preview_glass_closeup.png).

Why a glass sphere is in the scene at all: hogel-free holography and every
method it is compared against take colour-plus-depth input, and a depth map
stores one surface per pixel, so none of them can represent glass, smoke or a
mirror. Our pipeline consumes the light field directly and has no such limit.
The sphere is the test case that distinguishes the two.
"""

import math

import numpy as np
import pytest
import sympy as sp

from tier1_lightfield.cornell_lightfield import (
    MAT_GLASS,
    MATERIALS,
    build_scene,
    fresnel_reflectance,
    intersect_sphere,
    refract_dir,
)

GLASS_IOR = 1.5


def _random_incidence(rng):
    """Unit direction striking the +z plane from above, with the normal at +z."""
    v = rng.standard_normal(3)
    v[2] = -abs(v[2])
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_refraction_obeys_snells_law_on_random_incidences(seed):
    rng = np.random.default_rng(seed)
    worst_unit = worst_snell = 0.0
    transmitted = 0

    for _ in range(500):
        d = _random_incidence(rng)
        n1, n2 = rng.uniform(1.0, 2.0), rng.uniform(1.0, 2.0)
        ok, tx, ty, tz = refract_dir(d[0], d[1], d[2], 0.0, 0.0, 1.0, n1 / n2)
        if not ok:
            continue
        transmitted += 1
        t = np.array([tx, ty, tz])
        worst_unit = max(worst_unit, abs(np.linalg.norm(t) - 1.0))
        sin_i = math.hypot(d[0], d[1])
        sin_t = math.hypot(t[0], t[1])
        worst_snell = max(worst_snell, abs(n1 * sin_i - n2 * sin_t))

    assert transmitted > 200
    assert worst_unit < 1e-12
    assert worst_snell < 1e-12


def test_refracted_ray_never_emerges_on_the_incident_side():
    rng = np.random.default_rng(3)
    for _ in range(500):
        d = _random_incidence(rng)
        ok, tx, ty, tz = refract_dir(d[0], d[1], d[2], 0.0, 0.0, 1.0, 1.0 / GLASS_IOR)
        if ok:
            assert tz < 0.0


@pytest.mark.parametrize("n1,n2", [(1.0, 1.5), (1.5, 1.0), (1.0, 1.33)])
def test_fresnel_at_normal_incidence_matches_the_closed_form(n1, n2):
    assert fresnel_reflectance(1.0, n1, n2) == pytest.approx(
        ((n1 - n2) / (n1 + n2)) ** 2, rel=1e-12)


def test_fresnel_normal_incidence_closed_form_derived_symbolically():
    """Do not take the (n1-n2)^2/(n1+n2)^2 form on trust; derive it."""
    a, b = sp.symbols("n1 n2", positive=True)
    cos_t = sp.sqrt(1 - (a / b) ** 2 * (1 - 1 ** 2))
    rs = (a * 1 - b * cos_t) / (a * 1 + b * cos_t)
    rp = (a * cos_t - b * 1) / (a * cos_t + b * 1)

    assert sp.simplify((rs ** 2 + rp ** 2) / 2 - ((a - b) / (a + b)) ** 2) == 0


def test_fresnel_tends_to_total_reflection_at_grazing_incidence():
    previous = 0.0
    for cos_i in (1e-1, 1e-2, 1e-4, 1e-6):
        r = fresnel_reflectance(cos_i, 1.0, GLASS_IOR)
        assert r > previous
        previous = r
    assert previous == pytest.approx(1.0, abs=1e-5)


def test_p_polarisation_vanishes_exactly_at_the_brewster_angle():
    n1, n2 = 1.0, GLASS_IOR
    cos_b = math.cos(math.atan(n2 / n1))
    cos_t = math.sqrt(1 - (n1 / n2) ** 2 * (1 - cos_b ** 2))
    rp = (n1 * cos_t - n2 * cos_b) / (n1 * cos_t + n2 * cos_b)

    assert rp == pytest.approx(0.0, abs=1e-15)


def test_total_internal_reflection_switches_on_at_the_critical_angle():
    critical = math.asin(1.0 / GLASS_IOR)

    assert fresnel_reflectance(math.cos(critical - 1e-3), GLASS_IOR, 1.0) < 0.99
    assert fresnel_reflectance(math.cos(critical + 1e-3), GLASS_IOR, 1.0) == 1.0

    ok, _, _, _ = refract_dir(
        math.sin(critical + 1e-3), 0.0, -math.cos(critical + 1e-3),
        0.0, 0.0, 1.0, GLASS_IOR)
    assert not ok


def test_sphere_intersection_distances():
    # unit sphere at the origin, ray along +x from x = -3
    assert intersect_sphere(-3.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0) == pytest.approx(2.0)
    # a miss passes by at y = 2
    assert intersect_sphere(-3.0, 2.0, 0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0) == -1.0
    # from inside, the near root is behind us so the far one must be returned
    assert intersect_sphere(0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)
    # grazing the surface exactly
    assert intersect_sphere(-3.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 1.0) == pytest.approx(3.0)


def test_the_sphere_rests_on_the_small_box_without_intersecting_it():
    _, _, boxes, _, spheres, sphere_mats, _ = build_scene()
    cx, cy, cz, r = spheres[0]
    small = boxes[1]

    assert sphere_mats[0] == MAT_GLASS
    assert MATERIALS[MAT_GLASS, 6] == pytest.approx(GLASS_IOR)

    # sits exactly on the lid, not sunk into it and not floating
    assert cy - r == pytest.approx(small[4], abs=1e-12)
    # centred over the lid in both horizontal directions
    assert cx == pytest.approx(0.5 * (small[0] + small[3]), abs=1e-12)
    assert cz == pytest.approx(0.5 * (small[2] + small[5]), abs=1e-12)
    # and it does not overhang
    assert r <= 0.5 * (small[3] - small[0]) + 1e-12


def test_glass_neither_absorbs_nor_emits():
    """Clear glass: any albedo or emission on it would be a modelling error."""
    assert MATERIALS[MAT_GLASS, 3:6] == pytest.approx(0.0)
    assert MATERIALS[MAT_GLASS, 0:3] == pytest.approx(1.0)
