"""Statistics of the Trajectron++ output distribution, straight from the decoder's own
Gaussian-mixture parameters.

What the decoder actually emits
-------------------------------
For an agent at timestep `t` the model enumerates all K = 25 values of the discrete latent
`z` (`full_dist=True`), decodes each one, and integrates it through the agent's dynamics
(Unicycle for vehicles). The result stored in the bundle is, per horizon step h:

    dist_pis[h]  (K,)      p(z_k | history, neighbours, map) -- the latent posterior
    dist_mus[h]  (K, 2)    mean position of mode k at h
    dist_covs[h] (K, 2, 2) covariance of mode k at h

so the predicted position at h is the mixture  p(x_h) = sum_k pi_k * N(mu_k[h], Sigma_k[h]).
`GMM_components: 1` in the checkpoint means each latent mode decodes to *one* bivariate
Gaussian; the mixture is over the latent, not on top of a second per-mode mixture. The
weights are a function of the encoder alone, so `dist_pis` is identical at every h -- the
horizon changes where the modes go, never how much they are believed.

The decomposition this module is built around
---------------------------------------------
Moment-matching the mixture at each h gives one Gaussian with

    mu  = sum_k pi_k mu_k
    C   = sum_k pi_k Sigma_k             (WITHIN : how uncertain each mode is on its own)
        + sum_k pi_k (mu_k-mu)(mu_k-mu)^T (BETWEEN: how far the modes disagree)

and since trace is linear, trace(C) = trace(W) + trace(B) exactly, i.e. the RMS radii add in
quadrature. That splits "the uncertainty grows over the horizon" into the two things it can
mean -- one blurry intention, or several sharp intentions pulling apart -- which is the whole
point of looking at the mixture rather than at a sample cloud.

The GMM and the samples are NOT the same object
-----------------------------------------------
`Unicycle.integrate_distribution` rolls the *mean* control through the exact nonlinear
dynamics and the covariance through an EKF linearization, C_{t+1} = F C F' + G Q G'. The
sampling pass instead draws controls and integrates each one exactly. Three consequences,
all of which this module measures rather than assumes (`sigma_samples`, `mean_gap`):

  * mu_k is f(E[u]), not E[f(u)] -- for a turning vehicle those differ, so the drawn mean
    path is not the mean of the sample fan;
  * the ellipse is a first-order approximation of a nonlinear pushforward, exact only while
    the horizon is short relative to the curvature;
  * the control clamps added to `Unicycle.dynamic` (max_a / max_heading_change) bound every
    sample but nothing in the covariance recursion -- `compute_jacobian` and
    `compute_control_jacobian` are still evaluated at the raw mean control -- so the ellipse
    can be wider than any trajectory the sampler is able to produce, and mean and covariance
    are not linearized about quite the same trajectory when a mean control is out of bounds.

So a Gaussian blob is the model's linearized belief, and the sample fan is its true
pushforward. Where they disagree, the disagreement is the finding.

Everything here is numpy-only and coordinate-frame agnostic: pass positions in whatever frame
you want the answers in (the bundle's scene-local, or world after adding meta x_min/y_min --
covariances are the same in both, since the two differ by a translation).
"""
import numpy as np


def mixture_moments(mus, covs, pis):
    """Moment-match the per-step mixture. Returns (mean, cov, within, between).

    :param mus:  (ph, K, 2) component means
    :param covs: (ph, K, 2, 2) component covariances
    :param pis:  (ph, K) mixture weights (each row sums to 1)
    :returns: mean (ph, 2), cov (ph, 2, 2), within (ph, 2, 2), between (ph, 2, 2);
        cov == within + between to machine precision.
    """
    mus, covs, pis = np.asarray(mus, float), np.asarray(covs, float), np.asarray(pis, float)
    mean = np.einsum('hk,hkd->hd', pis, mus)
    d = mus - mean[:, None, :]
    within = np.einsum('hk,hkij->hij', pis, covs)
    between = np.einsum('hk,hki,hkj->hij', pis, d, d)
    return mean, within + between, within, between


def rms_radius(cov):
    """sqrt(trace(cov)) per step: the RMS distance from the mean, in metres.

    Chosen over a per-axis sigma because trace is linear, so the within/between split adds in
    quadrature exactly: rms(C)^2 = rms(W)^2 + rms(B)^2.
    """
    return np.sqrt(np.trace(np.asarray(cov, float), axis1=-2, axis2=-1))


def axis_sigmas(cov, u):
    """(sigma_along, sigma_across) of each step's covariance in the frame of unit vector `u`.

    For a vehicle `u` is its current heading, which separates the two very different things
    the model is unsure about: how far it will get (speed/longitudinal) versus whether it
    will leave its lane or turn (lateral).
    """
    cov = np.asarray(cov, float)
    u = np.asarray(u, float) / max(np.linalg.norm(u), 1e-9)
    v = np.array([-u[1], u[0]])
    return (np.sqrt(np.einsum('i,hij,j->h', u, cov, u)),
            np.sqrt(np.einsum('i,hij,j->h', v, cov, v)))


def effective_modes(pis):
    """exp(H(pi)) per step -- the perplexity of the latent posterior.

    1 = the encoder committed to a single latent mode; K = it is uniform over all of them.
    A useful sanity number: a large `between` term with perplexity ~1 would be contradictory.
    """
    p = np.asarray(pis, float)
    return np.exp(-np.sum(p * np.log(np.clip(p, 1e-12, None)), axis=-1))


def mahalanobis(points, mean, cov):
    """Mahalanobis distance of `points` (ph, 2) under the moment-matched Gaussian per step.

    Reported for the logged ground truth: it says how many sigmas out the truth landed, which
    the raw displacement error cannot, because a 5 m error under a 10 m spread and the same
    error under a 1 m spread are not the same miss.
    """
    d = np.asarray(points, float) - np.asarray(mean, float)
    out = np.full(len(d), np.nan)
    for h, (dh, ch) in enumerate(zip(d, np.asarray(cov, float))):
        try:
            out[h] = np.sqrt(max(dh @ np.linalg.solve(ch + 1e-9 * np.eye(2), dh), 0.0))
        except np.linalg.LinAlgError:
            pass
    return out


def heading_of(history):
    """Unit heading from the last two logged positions; +x if the agent has not moved."""
    h = np.asarray(history, float)
    if len(h) < 2:
        return np.array([1.0, 0.0])
    d = h[-1] - h[-2]
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([1.0, 0.0])


def speed_of(history, dt):
    """Speed at the last logged step, m/s."""
    h = np.asarray(history, float)
    return 0.0 if len(h) < 2 else float(np.linalg.norm(h[-1] - h[-2]) / dt)


def analyse_node(node, dt, offset=None):
    """All of the above for one bundle node record. Returns a dict of per-horizon arrays.

    Positions are shifted by `offset` (meta x_min/y_min) when given, so everything comes back
    in world/map coordinates; covariances are unaffected by the shift.

    Keys: pos, history, future, ml, mean, cov, within, between, sigma (rms of cov),
    sigma_within, sigma_between, sigma_long, sigma_lat, pis, perplexity, heading, speed,
    gt_error, gt_mahalanobis, samples, sample_mean, sigma_samples, mean_gap.
    """
    off = np.zeros(2) if offset is None else np.asarray(offset, float)
    hist = np.asarray(node['history'], float) + off
    fut = np.asarray(node['future'], float)
    fut = fut + off if fut.size else fut.reshape(0, 2)
    mus = np.asarray(node['dist_mus'], float)
    if mus.size == 0:
        raise ValueError(f"node {node['id']} carries no GMM -- the run that wrote this bundle "
                         f"was not launched with --style gaussian/both")
    mus = mus + off
    mean, cov, within, between = mixture_moments(mus, node['dist_covs'], node['dist_pis'])
    u = heading_of(hist)

    ph = len(mean)
    gt_err = np.full(ph, np.nan)
    gt_maha = np.full(ph, np.nan)
    if len(fut):
        n = min(ph, len(fut))
        gt_err[:n] = np.linalg.norm(fut[:n] - mean[:n], axis=1)
        gt_maha[:n] = mahalanobis(fut[:n], mean[:n], cov[:n])

    # The Monte Carlo pass, when the run stored one: same decoder, different integrator (see
    # the module docstring). Compared here per horizon step rather than assumed to agree.
    samples = np.asarray(node['samples'], float)
    sample_mean = np.full((ph, 2), np.nan)
    sigma_samples = np.full(ph, np.nan)
    mean_gap = np.full(ph, np.nan)
    if samples.size:
        samples = samples + off
        sample_mean = samples.mean(axis=0)
        d = samples - sample_mean
        sigma_samples = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=0))
        mean_gap = np.linalg.norm(sample_mean - mean, axis=1)

    sigma_long, sigma_lat = axis_sigmas(cov, u)
    return {
        'id': node['id'], 'pos': hist[-1], 'history': hist, 'future': fut,
        'ml': np.asarray(node['ml'], float) + off,
        'mus': mus, 'covs': np.asarray(node['dist_covs'], float),
        'pis': np.asarray(node['dist_pis'], float),
        'mean': mean, 'cov': cov, 'within': within, 'between': between,
        'sigma': rms_radius(cov), 'sigma_within': rms_radius(within),
        'sigma_between': rms_radius(between),
        'sigma_long': sigma_long, 'sigma_lat': sigma_lat,
        'perplexity': effective_modes(node['dist_pis']),
        'heading': u, 'speed': speed_of(hist, dt),
        'gt_error': gt_err, 'gt_mahalanobis': gt_maha,
        'samples': samples if samples.size else samples.reshape(0, ph, 2),
        'sample_mean': sample_mean, 'sigma_samples': sigma_samples, 'mean_gap': mean_gap,
    }
