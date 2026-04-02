"""
JAX-accelerated C_ell computation and binned log-likelihood for 3x2pt.

Python keeps: power spectra (cosmopower/bacco emulators), integrations and
interpolations that depend on scipy (e.g. get_ez_rz_k, kernel integrals).
JAX does: all sums and multiplications in C_ell integrands, z-integration
(trapezoidal rule), building the data vector, and chi2 with Cholesky solve.

Enable by setting use_jax: true in config (top-level or under theory), with
observable: '3x2pt' and likelihood: 'binned'. Requires: pip install jax jaxlib.

Why isn't the JAX path faster?
------------------------------
Almost all of the time per likelihood call is spent in *Python* inside
get_theory_arrays_3x2pt(): cosmopower/HMcode/bacco emulator calls, get_ez_rz_k
(quad over z), growth, kernels (simpson), Pmm/Pgm/Pgg, IA terms. The part JAX
accelerates (C_ell from those arrays + pack vector + Cholesky chi2) is only a
small fraction of the total (~few %). So end-to-end runtime is dominated by
the emulators and scipy, and JAX does not reduce it. The JAX path is mainly
useful for: (1) future gradient-based samplers if you expose d(log L)/d(params)
via the theory arrays, or (2) moving the C_ell+chi2 block to GPU if the rest
were also on GPU. For raw speed on clusters, optimize the Python side (e.g.
vectorize get_ez_rz_k, reduce emulator cost) rather than expecting gains from
this JAX layer.

Usage (optional, for custom runs):
    from MGLensing.jax_cell_like import build_log_likelihood_jax
    log_like_jax = build_log_likelihood_jax(use_cholesky=True)
    theory_arrays = Theory.get_theory_arrays_3x2pt(params)
    log_like = log_like_jax(theory_arrays, data_vector, cholesky_L, mask)
"""

import numpy as np

try:
    import jax.numpy as jnp
    from jax import jit
    from jax.scipy.integrate import trapezoid
    from jax.scipy.linalg import solve_triangular
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


if JAX_AVAILABLE:

    # Keys that are arrays (traced) vs static scalars (must be Python int/float for JIT).
    _ARRAY_KEYS = (
        'ez', 'rz', 'zz_integr', 'pmm', 'pgm', 'pgg',
        'w_gamma', 'w_ia', 'w_g', 'pk_delta_ia', 'pk_iaia', 'pk_gal_ia'
    )
    _STATIC_KEYS = (
        'nell_wl', 'nell_gc', 'nell_xc', 'nbin_s', 'nbin_l',
        'nbin_flat_s', 'nbin_flat_l', 'noise_LL', 'noise_GG', 'H0_h_c'
    )

    def _dict_to_jax(d):
        """Convert numpy arrays in dict to JAX arrays; scalars to Python int/float."""
        out = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                out[k] = jnp.array(v)
            elif isinstance(v, (np.integer, np.int32, np.int64)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, np.float32, np.float64)):
                out[k] = float(v)
            else:
                out[k] = v
        return out

    def _split_arrays_static(d):
        """Return (arrays_tuple, static_tuple). Static are Python int/float only (built here, not traced)."""
        ar_jax = _dict_to_jax(d)
        arrays = tuple(ar_jax[k] for k in _ARRAY_KEYS)
        static = []
        for k in _STATIC_KEYS:
            v = d[k]  # use original dict so we get Python scalars
            if isinstance(v, (np.integer, np.int32, np.int64)):
                static.append(int(v))
            elif isinstance(v, (np.floating, np.float32, np.float64)):
                static.append(float(v))
            elif k in ('nell_wl', 'nell_gc', 'nell_xc', 'nbin_s', 'nbin_l', 'nbin_flat_s', 'nbin_flat_l'):
                static.append(int(v))
            else:
                static.append(float(v) if isinstance(v, (int, float)) else v)
        return arrays, tuple(static)

    def _compute_cell_shear_jax(arrays, static):
        """C_ll from theory arrays. Integrand then trapezoid over z."""
        (ez, rz, zz, pmm, pgm, pgg, wg, wi, w_g, pk_delta_ia, pk_iaia, pk_gal_ia) = arrays
        nell_wl, nell_gc, nell_xc, nbin_s, nbin_l, _, _, noise_LL, noise_GG, H0_h_c = static
        kernel_wl = wg[:, :, :, None] * wg[:, :, None, :] * pmm[:, :, None, None]
        kernel_delta_ia = (wg[:, :, :, None] * wi[:, :, None, :] + wg[:, :, None, :] * wi[:, :, :, None]) * pk_delta_ia[:, :, None, None]
        kernel_iaia = wi[:, :, :, None] * wi[:, :, None, :] * pk_iaia[:, :, None, None]
        denom = ez[None, :, None, None] * rz[None, :, None, None] * rz[None, :, None, None] * H0_h_c
        integrand = (kernel_wl + kernel_delta_ia + kernel_iaia) / denom
        cl_ll = trapezoid(integrand, zz, axis=1)
        cl_ll = cl_ll[:nell_wl, :, :]
        cl_ll = cl_ll + noise_LL * jnp.eye(nbin_s)[None, :, :]
        return cl_ll

    def _compute_cell_galclust_jax(arrays, static):
        """C_gg from theory arrays."""
        (ez, rz, zz, pmm, pgm, pgg, wg, wi, w_g, pk_delta_ia, pk_iaia, pk_gal_ia) = arrays
        nell_wl, nell_gc, nell_xc, nbin_s, nbin_l, _, _, noise_LL, noise_GG, H0_h_c = static
        integrand = w_g[:, :, :, None] * w_g[:, :, None, :] * pgg / (ez[None, :, None, None] * rz[None, :, None, None] * rz[None, :, None, None] * H0_h_c)
        cl_gg = trapezoid(integrand, zz, axis=1)
        cl_gg = cl_gg[:nell_gc, :, :]
        cl_gg = cl_gg + noise_GG * jnp.eye(nbin_l)[None, :, :]
        return cl_gg

    def _compute_cell_cross_jax(arrays, static):
        """C_lg and C_gl from theory arrays."""
        (ez, rz, zz, pmm, pgm, pgg, w_gamma, w_ia, w_g, pk_delta_ia, pk_iaia, pk_gal_ia) = arrays
        nell_wl, nell_gc, nell_xc, nbin_s, nbin_l, _, _, noise_LL, noise_GG, H0_h_c = static
        kernel_wl_gal = w_gamma[:, :, :, None] * pgm[:, :, None, :]
        kernel_ia_gal = w_ia[:, :, :, None] * pk_gal_ia
        integrand = (kernel_wl_gal + kernel_ia_gal) * w_g[:, :, None, :] / (ez[None, :, None, None] * rz[None, :, None, None] * rz[None, :, None, None] * H0_h_c)
        cl_lg = trapezoid(integrand, zz, axis=1)
        cl_lg = cl_lg[:nell_xc, :, :]
        cl_gl = jnp.transpose(cl_lg, (0, 2, 1))
        return cl_lg, cl_gl

    def _cells_to_data_vector_flat_jax(cl_ll, cl_gg, cl_lg, cl_gl, static):
        """Pack C_ell into 3x2pt data vector (same layout as Theory.compute_data_vector_3x2pt)."""
        nell_wl, nell_gc, nell_xc, nbin_s, nbin_l, _, _, _, _, _ = static
        ll_blocks = []
        for bin1 in range(nbin_s):
            for bin2 in range(bin1, nbin_s):
                ll_blocks.append(cl_ll[:, bin1, bin2])
        vec_ll = jnp.concatenate(ll_blocks)
        xc_blocks = []
        for bin1 in range(nbin_l):
            for bin2 in range(nbin_s):
                xc_blocks.append(cl_gl[:, bin1, bin2])
        vec_xc = jnp.concatenate(xc_blocks)
        gg_blocks = []
        for bin1 in range(nbin_l):
            for bin2 in range(bin1, nbin_l):
                gg_blocks.append(cl_gg[:, bin1, bin2])
        vec_gg = jnp.concatenate(gg_blocks)
        return jnp.concatenate([vec_ll, vec_xc, vec_gg])

    @jit
    def _log_likelihood_binned_jax_impl(model_vector, data_vector, cholesky_L):
        """Chi2 = |L^{-1}(d-m)|^2, L lower."""
        diff = data_vector - model_vector
        y = solve_triangular(cholesky_L, diff, lower=True)
        chi2 = jnp.sum(jnp.square(y))
        return -0.5 * chi2

    @jit
    def _log_likelihood_binned_inv_cov_jax_impl(model_vector, data_vector, inv_cov):
        """Chi2 = (d-m)^T inv_cov (d-m)."""
        diff = data_vector - model_vector
        chi2 = jnp.dot(diff, jnp.dot(inv_cov, diff))
        return -0.5 * chi2

    @jit
    def compute_cells_and_model_vector_jax(arrays, static):
        """From (arrays_tuple, static_tuple), compute C_ell and packed 3x2pt model vector (full, unmasked)."""
        cl_ll = _compute_cell_shear_jax(arrays, static)
        cl_gg = _compute_cell_galclust_jax(arrays, static)
        cl_lg, cl_gl = _compute_cell_cross_jax(arrays, static)
        model_vector_full = _cells_to_data_vector_flat_jax(cl_ll, cl_gg, cl_lg, cl_gl, static)
        return model_vector_full

    def build_log_likelihood_jax(use_cholesky=True):
        """
        Build a function: (theory_arrays_numpy, data_vector, cov_info, mask) -> log_like.

        theory_arrays_numpy: dict from Theory.get_theory_arrays_3x2pt(params).
        data_vector: 1d array (masked length).
        cov_info: either cholesky_L (masked, lower) or inv_cov (masked).
        mask: boolean 1d, length = full data vector length; True where kept.

        Returns a function that converts theory arrays to JAX, computes model vector, applies mask,
        and returns -0.5*chi2. Use when likelihood == 'binned' and observable == '3x2pt'.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not installed. Install with: pip install jax jaxlib")

        _jit_cache = {}

        def run(theory_arrays_numpy, data_vector, cov_info, mask):
            arrays, static = _split_arrays_static(theory_arrays_numpy)
            nell_wl, nell_gc, nell_xc, nbin_s, nbin_l, nbin_flat_s, nbin_flat_l, noise_LL, noise_GG, H0_h_c = static
            mask = np.asarray(mask)
            indices = np.where(mask)[0].astype(np.int32)
            key = (static, tuple(indices))
            if key not in _jit_cache:
                indices_jax = jnp.array(indices)
                def log_like_impl(arrays, data_vector, cov_info):
                    (ez, rz, zz, pmm, pgm, pgg, wg, wi, w_g, pk_delta_ia, pk_iaia, pk_gal_ia) = arrays
                    denom = ez[None, :, None, None] * rz[None, :, None, None] * rz[None, :, None, None] * H0_h_c
                    kernel_wl = wg[:, :, :, None] * wg[:, :, None, :] * pmm[:, :, None, None]
                    k_dia = (wg[:, :, :, None] * wi[:, :, None, :] + wg[:, :, None, :] * wi[:, :, :, None]) * pk_delta_ia[:, :, None, None]
                    k_ia = wi[:, :, :, None] * wi[:, :, None, :] * pk_iaia[:, :, None, None]
                    cl_ll = trapezoid((kernel_wl + k_dia + k_ia) / denom, zz, axis=1)[:nell_wl, :, :] + noise_LL * jnp.eye(nbin_s)[None, :, :]
                    integ_gg = w_g[:, :, :, None] * w_g[:, :, None, :] * pgg / denom
                    cl_gg = trapezoid(integ_gg, zz, axis=1)[:nell_gc, :, :] + noise_GG * jnp.eye(nbin_l)[None, :, :]
                    k_wl_gal = wg[:, :, :, None] * pgm[:, :, None, :]
                    k_ia_gal = wi[:, :, :, None] * pk_gal_ia
                    cl_lg = trapezoid((k_wl_gal + k_ia_gal) * w_g[:, :, None, :] / denom, zz, axis=1)[:nell_xc, :, :]
                    cl_gl = jnp.transpose(cl_lg, (0, 2, 1))
                    ll_b = [cl_ll[:, b1, b2] for b1 in range(nbin_s) for b2 in range(b1, nbin_s)]
                    xc_b = [cl_gl[:, b1, b2] for b1 in range(nbin_l) for b2 in range(nbin_s)]
                    gg_b = [cl_gg[:, b1, b2] for b1 in range(nbin_l) for b2 in range(b1, nbin_l)]
                    model_full = jnp.concatenate([jnp.concatenate(ll_b), jnp.concatenate(xc_b), jnp.concatenate(gg_b)])
                    model_masked = model_full[indices_jax]
                    diff = data_vector - model_masked
                    if use_cholesky:
                        y = solve_triangular(cov_info, diff, lower=True)
                        return -0.5 * jnp.sum(jnp.square(y))
                    return -0.5 * jnp.dot(diff, jnp.dot(cov_info, diff))
                _jit_cache[key] = jit(log_like_impl)
            data_jax = jnp.array(data_vector)
            cov_jax = jnp.array(cov_info)
            return float(_jit_cache[key](arrays, data_jax, cov_jax))

        return run

else:
    build_log_likelihood_jax = None
    compute_cells_and_model_vector_jax = None
