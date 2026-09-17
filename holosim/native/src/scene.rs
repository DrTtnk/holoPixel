use napi_derive::napi;

// ── Vector Math ───────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let l = self.length();
        if l < 1e-12 { self } else { self.scale(1.0 / l) }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }
}

// ── Ray ───────────────────────────────────────────────────

pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

// ── Material ──────────────────────────────────────────────

#[derive(Clone, Copy)]
pub enum MaterialKind {
    Diffuse,
    Emissive,
}

#[derive(Clone, Copy)]
pub struct Material {
    pub albedo: Vec3,
    pub emission: Vec3,
    pub kind: MaterialKind,
}

impl Material {
    pub const fn diffuse(r: f32, g: f32, b: f32) -> Self {
        Self {
            albedo: Vec3::new(r, g, b),
            emission: Vec3::new(0.0, 0.0, 0.0),
            kind: MaterialKind::Diffuse,
        }
    }

    pub const fn emissive(r: f32, g: f32, b: f32, intensity: f32) -> Self {
        Self {
            albedo: Vec3::new(0.0, 0.0, 0.0),
            emission: Vec3::new(r * intensity, g * intensity, b * intensity),
            kind: MaterialKind::Emissive,
        }
    }
}

// ── Triangle ──────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct Triangle {
    pub v0: Vec3,
    pub v1: Vec3,
    pub v2: Vec3,
    pub normal: Vec3,
    pub material_idx: usize,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material_idx: usize) -> Self {
        let edge1 = v1.sub(v0);
        let edge2 = v2.sub(v0);
        let normal = edge1.cross(edge2).normalize();
        Self { v0, v1, v2, normal, material_idx }
    }
}

// ── Hit Record ────────────────────────────────────────────

pub struct Hit {
    pub t: f32,
    pub point: Vec3,
    pub normal: Vec3,
    pub material_idx: usize,
}

// ── Möller–Trumbore Ray-Triangle Intersection ─────────────

pub fn intersect_triangle(ray: &Ray, tri: &Triangle) -> Option<f32> {
    let edge1 = tri.v1.sub(tri.v0);
    let edge2 = tri.v2.sub(tri.v0);
    let h = ray.dir.cross(edge2);
    let a = edge1.dot(h);

    if a.abs() < 1e-7 { return None; }

    let f = 1.0 / a;
    let s = ray.origin.sub(tri.v0);
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) { return None; }

    let q = s.cross(edge1);
    let v = f * ray.dir.dot(q);
    if v < 0.0 || u + v > 1.0 { return None; }

    let t = f * edge2.dot(q);
    if t > 1e-4 { Some(t) } else { None }
}

// ── Scene ─────────────────────────────────────────────────

pub struct Scene {
    pub triangles: Vec<Triangle>,
    pub materials: Vec<Material>,
}

impl Scene {
    pub fn trace(&self, ray: &Ray) -> Option<Hit> {
        let mut closest: Option<Hit> = None;
        let mut best_t = f32::INFINITY;

        for tri in &self.triangles {
            if let Some(t) = intersect_triangle(ray, tri) {
                if t < best_t {
                    best_t = t;
                    let point = ray.origin + ray.dir.scale(t);
                    // Ensure normal faces the ray origin
                    let n = if tri.normal.dot(ray.dir) < 0.0 {
                        tri.normal
                    } else {
                        tri.normal.scale(-1.0)
                    };
                    closest = Some(Hit {
                        t,
                        point,
                        normal: n,
                        material_idx: tri.material_idx,
                    });
                }
            }
        }
        closest
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }
}

// ── Cornell Box Builder ───────────────────────────────────
// Standard Cornell Box: 552.8 units wide/deep, 548.8 tall.
// All coordinates in world space; we scale to unit cube later if needed.

pub fn build_cornell_box() -> Scene {
    let materials = vec![
        // 0: White diffuse (floor, ceiling, back wall)
        Material::diffuse(0.73, 0.73, 0.73),
        // 1: Red diffuse (left wall)
        Material::diffuse(0.63, 0.065, 0.05),
        // 2: Green diffuse (right wall)
        Material::diffuse(0.14, 0.45, 0.091),
        // 3: Area light
        Material::emissive(1.0, 1.0, 1.0, 40.0),
    ];

    let mut tris = Vec::with_capacity(36);

    // Helper: push a quad as two triangles (CCW winding from inside)
    let mut quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3, mat: usize| {
        tris.push(Triangle::new(a, b, c, mat));
        tris.push(Triangle::new(a, c, d, mat));
    };

    // Dimensions (standard Cornell Box, units ≈ mm)
    let w = 552.8_f32;  // width (x)
    let h = 548.8_f32;  // height (y)
    let d = 559.2_f32;  // depth (z)

    // Floor (y=0) — white
    quad(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(w, 0.0, 0.0),
        Vec3::new(w, 0.0, d),
        Vec3::new(0.0, 0.0, d),
        0,
    );

    // Ceiling (y=h) — white
    quad(
        Vec3::new(0.0, h, 0.0),
        Vec3::new(0.0, h, d),
        Vec3::new(w, h, d),
        Vec3::new(w, h, 0.0),
        0,
    );

    // Back wall (z=d) — white
    quad(
        Vec3::new(0.0, 0.0, d),
        Vec3::new(w, 0.0, d),
        Vec3::new(w, h, d),
        Vec3::new(0.0, h, d),
        0,
    );

    // Left wall (x=0) — red
    quad(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, d),
        Vec3::new(0.0, h, d),
        Vec3::new(0.0, h, 0.0),
        1,
    );

    // Right wall (x=w) — green
    quad(
        Vec3::new(w, 0.0, 0.0),
        Vec3::new(w, h, 0.0),
        Vec3::new(w, h, d),
        Vec3::new(w, 0.0, d),
        2,
    );

    // ── Tall box (left-back) ──────────────────────────────
    // Rotated ~17° CW around Y. Standard Cornell Box vertices:
    let tb = [
        Vec3::new(130.0, 0.0, 65.0),   // 0: front-left bottom
        Vec3::new(82.0, 0.0, 225.0),    // 1: back-left bottom
        Vec3::new(240.0, 0.0, 272.0),   // 2: back-right bottom
        Vec3::new(290.0, 0.0, 114.0),   // 3: front-right bottom
    ];
    let th = 330.0_f32; // tall box height
    let tb_top = [
        Vec3::new(tb[0].x, th, tb[0].z),
        Vec3::new(tb[1].x, th, tb[1].z),
        Vec3::new(tb[2].x, th, tb[2].z),
        Vec3::new(tb[3].x, th, tb[3].z),
    ];

    // Top face
    quad(tb_top[0], tb_top[3], tb_top[2], tb_top[1], 0);
    // Front face
    quad(tb[0], tb[3], tb_top[3], tb_top[0], 0);
    // Right face
    quad(tb[3], tb[2], tb_top[2], tb_top[3], 0);
    // Back face
    quad(tb[2], tb[1], tb_top[1], tb_top[2], 0);
    // Left face
    quad(tb[1], tb[0], tb_top[0], tb_top[1], 0);

    // ── Short box (right-front) ───────────────────────────
    let sb = [
        Vec3::new(265.0, 0.0, 296.0),   // 0: front-left bottom
        Vec3::new(314.0, 0.0, 456.0),   // 1: back-left bottom
        Vec3::new(472.0, 0.0, 406.0),   // 2: back-right bottom
        Vec3::new(423.0, 0.0, 247.0),   // 3: front-right bottom
    ];
    let sh = 165.0_f32; // short box height
    let sb_top = [
        Vec3::new(sb[0].x, sh, sb[0].z),
        Vec3::new(sb[1].x, sh, sb[1].z),
        Vec3::new(sb[2].x, sh, sb[2].z),
        Vec3::new(sb[3].x, sh, sb[3].z),
    ];

    // Top face
    quad(sb_top[0], sb_top[3], sb_top[2], sb_top[1], 0);
    // Front face
    quad(sb[0], sb[3], sb_top[3], sb_top[0], 0);
    // Right face
    quad(sb[3], sb[2], sb_top[2], sb_top[3], 0);
    // Back face
    quad(sb[2], sb[1], sb_top[1], sb_top[2], 0);
    // Left face
    quad(sb[1], sb[0], sb_top[0], sb_top[1], 0);

    // ── Area Light (ceiling, centered) ────────────────────
    let lx0 = 213.0_f32;
    let lx1 = 343.0_f32;
    let lz0 = 227.0_f32;
    let lz1 = 332.0_f32;
    let ly = h - 0.01; // slightly below ceiling to avoid z-fight

    quad(
        Vec3::new(lx0, ly, lz0),
        Vec3::new(lx1, ly, lz0),
        Vec3::new(lx1, ly, lz1),
        Vec3::new(lx0, ly, lz1),
        3,
    );

    Scene { triangles: tris, materials }
}

// ── napi exports ──────────────────────────────────────────

#[napi(object)]
pub struct SceneInfo {
    pub triangle_count: u32,
    pub material_count: u32,
    pub scene_name: String,
}

#[napi]
pub fn get_cornell_box_info() -> SceneInfo {
    let scene = build_cornell_box();
    SceneInfo {
        triangle_count: scene.triangle_count() as u32,
        material_count: scene.material_count() as u32,
        scene_name: "Cornell Box".to_string(),
    }
}

// Quick sanity: trace a single ray down the center and report what it hits
#[napi(object)]
pub struct TraceResult {
    pub hit: bool,
    pub distance: f64,
    pub normal: Vec<f64>,
    pub material_idx: u32,
    pub material_albedo: Vec<f64>,
    pub is_emissive: bool,
}

#[napi]
pub fn trace_test_ray() -> TraceResult {
    let scene = build_cornell_box();

    // Shoot ray from camera position toward center of back wall
    let ray = Ray {
        origin: Vec3::new(278.0, 273.0, -800.0),
        dir: Vec3::new(0.0, 0.0, 1.0),
    };

    match scene.trace(&ray) {
        Some(hit) => {
            let mat = &scene.materials[hit.material_idx];
            let emissive = matches!(mat.kind, MaterialKind::Emissive);
            TraceResult {
                hit: true,
                distance: hit.t as f64,
                normal: vec![hit.normal.x as f64, hit.normal.y as f64, hit.normal.z as f64],
                material_idx: hit.material_idx as u32,
                material_albedo: vec![mat.albedo.x as f64, mat.albedo.y as f64, mat.albedo.z as f64],
                is_emissive: emissive,
            }
        }
        None => TraceResult {
            hit: false,
            distance: 0.0,
            normal: vec![0.0, 0.0, 0.0],
            material_idx: 0,
            material_albedo: vec![0.0, 0.0, 0.0],
            is_emissive: false,
        },
    }
}

// ── Unit Tests ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Vec3 math ---

    #[test]
    fn vec3_dot_orthogonal() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert!((a.dot(b)).abs() < 1e-6);
    }

    #[test]
    fn vec3_dot_parallel() {
        let a = Vec3::new(3.0, 0.0, 0.0);
        assert!((a.dot(a) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_cross_product() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-6);
        assert!((z.y).abs() < 1e-6);
        assert!((z.z - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert!((n.x - 0.6).abs() < 1e-6);
        assert!((n.y - 0.8).abs() < 1e-6);
    }

    #[test]
    fn vec3_normalize_zero() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        let n = v.normalize();
        // Should not crash, returns zero vector
        assert!(n.length() < 1e-6);
    }

    // --- Cornell Box scene ---

    #[test]
    fn cornell_box_triangle_count() {
        let scene = build_cornell_box();
        // 5 walls × 2 + 2 boxes × 5 faces × 2 + 1 light × 2 = 32
        assert_eq!(scene.triangle_count(), 32);
    }

    #[test]
    fn cornell_box_material_count() {
        let scene = build_cornell_box();
        assert_eq!(scene.material_count(), 4); // white, red, green, emissive
    }

    // --- Ray-triangle intersection ---

    #[test]
    fn intersect_simple_triangle() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, 5.0),
            Vec3::new(1.0, -1.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
            0,
        );
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        };
        let t = intersect_triangle(&ray, &tri);
        assert!(t.is_some());
        assert!((t.unwrap() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn intersect_miss_triangle() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, 5.0),
            Vec3::new(1.0, -1.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
            0,
        );
        let ray = Ray {
            origin: Vec3::new(5.0, 5.0, 0.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        };
        assert!(intersect_triangle(&ray, &tri).is_none());
    }

    #[test]
    fn intersect_behind_ray_origin() {
        let tri = Triangle::new(
            Vec3::new(-1.0, -1.0, -5.0),
            Vec3::new(1.0, -1.0, -5.0),
            Vec3::new(0.0, 1.0, -5.0),
            0,
        );
        let ray = Ray {
            origin: Vec3::new(0.0, 0.0, 0.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        };
        // Triangle is behind the ray, should not hit
        assert!(intersect_triangle(&ray, &tri).is_none());
    }

    // --- Scene tracing ---

    #[test]
    fn trace_center_ray_hits_something() {
        let scene = build_cornell_box();
        let ray = Ray {
            origin: Vec3::new(278.0, 273.0, -800.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        };
        let hit = scene.trace(&ray);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        // Should hit back wall (z≈559.2, mat 0) or a box (mat 0, closer z).
        // Key: it hits something white (material 0) at positive t.
        assert!(hit.t > 0.0);
        assert_eq!(hit.material_idx, 0); // white diffuse
        // Hit z must be inside the box (0..559.2)
        assert!(hit.point.z >= 0.0 && hit.point.z <= 560.0);
    }

    #[test]
    fn trace_toward_left_wall_hits_red() {
        let scene = build_cornell_box();
        let ray = Ray {
            origin: Vec3::new(278.0, 273.0, 280.0),
            dir: Vec3::new(-1.0, 0.0, 0.0),
        };
        let hit = scene.trace(&ray);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.material_idx, 1); // red
    }

    #[test]
    fn trace_toward_right_wall_hits_green() {
        let scene = build_cornell_box();
        let ray = Ray {
            origin: Vec3::new(278.0, 273.0, 280.0),
            dir: Vec3::new(1.0, 0.0, 0.0),
        };
        let hit = scene.trace(&ray);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.material_idx, 2); // green
    }

    #[test]
    fn trace_upward_toward_light() {
        let scene = build_cornell_box();
        // From center of room, shoot straight up
        let ray = Ray {
            origin: Vec3::new(278.0, 100.0, 280.0),
            dir: Vec3::new(0.0, 1.0, 0.0),
        };
        let hit = scene.trace(&ray);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        // Should hit the ceiling area light (material 3) at y ≈ 548.8
        assert_eq!(hit.material_idx, 3);
        assert!((hit.point.y - 548.8).abs() < 1.0);
    }

    #[test]
    fn trace_escape_from_front() {
        let scene = build_cornell_box();
        // Shoot ray backward out of the open face (−z)
        let ray = Ray {
            origin: Vec3::new(278.0, 273.0, 10.0),
            dir: Vec3::new(0.0, 0.0, -1.0),
        };
        // Box is open at z=0 — no geometry there
        assert!(scene.trace(&ray).is_none());
    }

    // --- Triangle normal direction ---

    #[test]
    fn trace_normal_faces_ray() {
        let scene = build_cornell_box();
        let ray = Ray {
            origin: Vec3::new(278.0, 273.0, -800.0),
            dir: Vec3::new(0.0, 0.0, 1.0),
        };
        let hit = scene.trace(&ray).unwrap();
        // Normal should face the ray origin (dot(normal, dir) < 0)
        assert!(hit.normal.dot(ray.dir) < 0.0);
    }
}
