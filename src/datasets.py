"""
Benchmark dataset generators for the differentiation study.

Every generator returns a dict:
    {
        "name": str,
        "t": array (sample times, strictly increasing),
        "y": array (noisy/observed values),
        "dydt_true": array (ground-truth derivative at each t, EXACT,
                             known analytically or from the ODE RHS),
    }
"""

import numpy as np


RNG_SEED = 12345


def _uneven_grid(rng, t_min, t_max, n, jitter=0.6):
    """Strictly increasing but non-uniform sample times: start from a
    uniform grid and randomly perturb + sort, then also randomly stretch
    gaps so density varies across the domain."""
    base = np.linspace(t_min, t_max, n)
    step = (t_max - t_min) / (n - 1)
    perturb = rng.uniform(-jitter, jitter, size=n) * step
    t = base + perturb
    t = np.sort(t)
    t[0], t[-1] = t_min, t_max
    # enforce strictly increasing (resolve any accidental ties)
    for i in range(1, n):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-9
    return t


def _apply_missing(rng, t, y, dydt, frac):
    if frac <= 0:
        return t, y, dydt
    n = len(t)
    keep = rng.random(n) > frac
    keep[0] = keep[-1] = True
    return t[keep], y[keep], dydt[keep]


def _add_noise(rng, y, kind, level, outlier_frac=0.03, t=None, t_max=None):
    if kind == "none":
        return y
    if kind == "gaussian":
        return y + rng.normal(0, level, size=len(y))
    if kind == "gaussian_ramp":
        # heteroscedastic: sigma ramps linearly from 0.2*level to 1.8*level
        # across the time domain (V1.4 scenario)
        s = level * (0.2 + 1.6 * t / t_max)
        return y + rng.normal(0, 1, size=len(y)) * s
    if kind == "uniform":
        return y + rng.uniform(-level, level, size=len(y))
    if kind == "outliers":
        y2 = y.copy()
        n_out = max(1, int(outlier_frac * len(y)))
        idx = rng.choice(len(y), size=n_out, replace=False)
        y2[idx] += rng.choice([-1, 1], size=n_out) * rng.uniform(5, 15, size=n_out) * (
            level if level > 0 else np.std(y) * 0.5 + 1e-6
        )
        return y2
    raise ValueError(kind)


def make_dataset(kind, n=200, t_min=0.0, t_max=10.0, uneven=True,
                  noise="gaussian", noise_level=0.05, missing_frac=0.0,
                  outlier_frac=0.03, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    t = _uneven_grid(rng, t_min, t_max, n) if uneven else np.linspace(t_min, t_max, n)

    if kind == "constant_velocity":
        v = 2.3
        f = v * t
        fp = np.full_like(t, v)

    elif kind == "constant_acceleration":
        a, v0 = 1.7, 0.5
        f = 0.5 * a * t**2 + v0 * t
        fp = a * t + v0

    elif kind == "constant_jerk":
        j, a0, v0 = 0.9, -1.2, 2.0
        f = j / 6 * t**3 + a0 / 2 * t**2 + v0 * t
        fp = j / 2 * t**2 + a0 * t + v0

    elif kind == "polynomial":
        # quartic: f(t) = c4 t^4 + c3 t^3 + c2 t^2 + c1 t + c0
        c = [0.3, -1.1, 0.6, 2.0, -0.4]
        f = c[0] * t**4 + c[1] * t**3 + c[2] * t**2 + c[3] * t + c[4]
        fp = 4 * c[0] * t**3 + 3 * c[1] * t**2 + 2 * c[2] * t + c[3]

    elif kind == "sinusoidal":
        A, w, phi = 1.5, 1.8, 0.4
        f = A * np.sin(w * t + phi)
        fp = A * w * np.cos(w * t + phi)

    elif kind == "exponential":
        A, k = 0.8, 0.35
        f = A * np.exp(k * t)
        fp = A * k * np.exp(k * t)

    elif kind == "projectile":
        v0, theta, g = 12.0, np.deg2rad(50), 9.81
        f = v0 * np.sin(theta) * t - 0.5 * g * t**2
        fp = v0 * np.sin(theta) - g * t

    elif kind == "circular":
        R, w = 3.0, 1.2
        f = R * np.cos(w * t)
        fp = -R * w * np.sin(w * t)

    elif kind == "chaotic":
        f, fp = _lorenz_x(t)

    else:
        raise ValueError(kind)

    y = _add_noise(rng, f, noise, noise_level, outlier_frac=outlier_frac,
                   t=t, t_max=t_max)
    t, y, fp = _apply_missing(rng, t, y, fp, missing_frac)

    return {"name": kind, "t": t, "y": y, "dydt_true": fp}


def _lorenz_x(t_query):
    """Integrate the Lorenz system with scipy and return x(t) and the
    EXACT derivative dx/dt at the queried times, obtained directly from
    the ODE right-hand side (sigma*(y-x)) evaluated on the integrated
    trajectory -- not from finite differencing the trajectory."""
    from scipy.integrate import solve_ivp

    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(t, s):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    t0, t1 = float(t_query[0]), float(t_query[-1])
    shift = 5.0  # shift so trajectory has settled a bit before t0 (avoid transient at 0)
    sol = solve_ivp(rhs, (0, t1 + shift + 1), [1.0, 1.0, 1.0], dense_output=True,
                     max_step=0.01, rtol=1e-10, atol=1e-12)
    s = sol.sol(t_query + shift)
    x, y, z = s
    xp = sigma * (y - x)
    return x, xp


def all_dataset_specs():
    """Returns the list of (kind, kwargs) combinations used in the full
    benchmark suite, covering the required scenario matrix."""
    specs = []
    motions = ["constant_velocity", "constant_acceleration", "constant_jerk",
               "polynomial", "sinusoidal", "exponential", "projectile",
               "circular", "chaotic"]

    for m in motions:
        specs.append((f"{m}__clean_uniform", dict(
            kind=m, uneven=False, noise="none", noise_level=0.0)))
        specs.append((f"{m}__uneven_clean", dict(
            kind=m, uneven=True, noise="none", noise_level=0.0)))
        specs.append((f"{m}__uneven_gaussian", dict(
            kind=m, uneven=True, noise="gaussian", noise_level=0.05)))
        specs.append((f"{m}__uneven_uniformnoise", dict(
            kind=m, uneven=True, noise="uniform", noise_level=0.05)))
        specs.append((f"{m}__uneven_outliers", dict(
            kind=m, uneven=True, noise="outliers", noise_level=0.0)))
        specs.append((f"{m}__missing_noisy", dict(
            kind=m, uneven=True, noise="gaussian", noise_level=0.05,
            missing_frac=0.25)))

    # a couple of "real-world-like" composite-difficulty cases
    specs.append(("sinusoidal__real_world_like", dict(
        kind="sinusoidal", uneven=True, noise="gaussian", noise_level=0.08,
        missing_frac=0.15, n=150)))
    specs.append(("chaotic__real_world_like", dict(
        kind="chaotic", uneven=True, noise="gaussian", noise_level=0.02,
        missing_frac=0.1, n=400, t_min=0.0, t_max=8.0)))

    # V1.3 contamination sweep: heavier gross-outlier fractions (the
    # original uneven_outliers scenario stays at 3% for continuity with
    # V1.0-V1.2). Appended AFTER the original 35 scenarios so earlier
    # per-dataset rows remain directly comparable across versions.
    for m in motions:
        specs.append((f"{m}__outliers10", dict(
            kind=m, uneven=True, noise="outliers", noise_level=0.0,
            outlier_frac=0.10)))
        specs.append((f"{m}__outliers25", dict(
            kind=m, uneven=True, noise="outliers", noise_level=0.0,
            outlier_frac=0.25)))

    # V1.4 heteroscedastic scenarios: noise sigma ramps 0.2x -> 1.8x of
    # level across the domain. Appended after everything above so all
    # earlier per-dataset rows stay directly comparable across versions.
    for m in motions:
        specs.append((f"{m}__hetero", dict(
            kind=m, uneven=True, noise="gaussian_ramp", noise_level=0.08)))

    return specs
