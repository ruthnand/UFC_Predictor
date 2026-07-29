"""Glicko-2 rating system (Glickman, 2013), applied one fight per rating period.

Unlike Elo, Glicko-2 tracks a rating deviation (RD, the uncertainty around the
rating) and a volatility. New fighters carry wide RDs so early results move them
quickly, while established fighters are more stable; RD also grows with
inactivity, which suits fighters returning from long layoffs.
"""

import math

SCALE = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0
DEFAULT_VOL = 0.06
TAU = 0.5
_EPS = 1e-6


def new_rating():
    return {"rating": DEFAULT_RATING, "rd": DEFAULT_RD, "vol": DEFAULT_VOL}


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi ** 2)


def expected_score(rating_a, rating_b):
    """Win probability of A over B, accounting for both fighters' uncertainty."""
    mu_a = (rating_a["rating"] - DEFAULT_RATING) / SCALE
    mu_b = (rating_b["rating"] - DEFAULT_RATING) / SCALE
    phi = math.sqrt((rating_a["rd"] / SCALE) ** 2 + (rating_b["rd"] / SCALE) ** 2)
    return 1.0 / (1.0 + math.exp(-_g(phi) * (mu_a - mu_b)))


def _update_one(rating, opp, score):
    """Return the updated rating dict for one game against ``opp``."""
    mu = (rating["rating"] - DEFAULT_RATING) / SCALE
    phi = rating["rd"] / SCALE
    sigma = rating["vol"]
    mu_j = (opp["rating"] - DEFAULT_RATING) / SCALE
    phi_j = opp["rd"] / SCALE

    g_j = _g(phi_j)
    e_j = 1.0 / (1.0 + math.exp(-g_j * (mu - mu_j)))
    v = 1.0 / (g_j * g_j * e_j * (1.0 - e_j))
    delta = v * g_j * (score - e_j)

    # Volatility update via the Illinois algorithm.
    a = math.log(sigma * sigma)

    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        big_b = a - k * TAU

    f_a, f_b = f(big_a), f(big_b)
    while abs(big_b - big_a) > _EPS:
        big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a)
        f_c = f(big_c)
        if f_c * f_b <= 0:
            big_a, f_a = big_b, f_b
        else:
            f_a /= 2.0
        big_b, f_b = big_c, f_c
    sigma_new = math.exp(big_a / 2.0)

    phi_star = math.sqrt(phi * phi + sigma_new * sigma_new)
    phi_new = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu_new = mu + phi_new * phi_new * g_j * (score - e_j)

    return {
        "rating": DEFAULT_RATING + SCALE * mu_new,
        "rd": SCALE * phi_new,
        "vol": sigma_new,
    }


def update_pair(winner, loser):
    """Update both fighters after a bout; returns (new_winner, new_loser).
    Computed from each side's pre-fight ratings, so order doesn't leak."""
    return _update_one(winner, loser, 1.0), _update_one(loser, winner, 0.0)
