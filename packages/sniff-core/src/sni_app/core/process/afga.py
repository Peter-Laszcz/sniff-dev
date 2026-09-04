# Spectroscopic Neutron Imaging Full-processing Framework (SNIFF)
# Copyright (C) 2026  ISIS Neutron and Muon Source
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code was primarily developed by Peter Laszcz.
# They can be contacted via laszczpeter@gmail.com.

"""
AFGA port for Simulation tab.
Computes microscopic cross-section over wavelength.
"""

from typing import Tuple

import NCrystal as NC
import numpy as np

_ENERGY_MIN_EV = 0.000818
_ENERGY_MAX_EV = 0.32725
_ENERGY_POINTS = 10000

SPEC_BUILDING_BLOCKS: Tuple[str, ...] = (
    "CH3",
    "CH2",
    "CHali",
    "CHaro",
    "OH",
    "NH2",
    "SH",
    "NH",
    "NH3",
)
"""
HFG hydrogen-group building blocks offered by the Simulation tab's spec builder,
in button order.  A "spec" is a "+"-joined list of these, each optionally
prefixed with a multiplier, e.g. "2xCH3+CHali".
"""


def energy_grid(
    points: int = _ENERGY_POINTS,
    energy_min: float = _ENERGY_MIN_EV,
    energy_max: float = _ENERGY_MAX_EV,
) -> np.ndarray:
    """
    Return the energy grid in eV.

    Parameters
    ----------
    points : int
        Number of grid points.
    energy_min, energy_max : float
        Grid bounds in eV. Both must be positive.
    """
    if energy_min <= 0 or energy_max <= 0:
        raise ValueError("Energy bounds must be positive.")
    return np.geomspace(energy_min, energy_max, points)


_DE_BROGLIE_A_SQRT_MEV = 9.045 #De Broglie constant


def wavelengths(energy_ev: np.ndarray) -> np.ndarray:
    """Convert neutron kinetic energy (eV) to wavelength (angstroms)."""
    energy_mev = np.asarray(energy_ev, dtype=float) * 1000.0
    return _DE_BROGLIE_A_SQRT_MEV / np.sqrt(energy_mev)


def energies(wavelength_a: np.ndarray) -> np.ndarray:
    """Convert neutron wavelength (angstroms) to kinetic energy (eV)."""
    energy_mev = (_DE_BROGLIE_A_SQRT_MEV / np.asarray(wavelength_a, dtype=float)) ** 2
    return energy_mev / 1000.0


WAVELENGTH_MIN_A = float(wavelengths(_ENERGY_MAX_EV))
"""
Shortest wavelength of the default range, in Angstroms.
"""

WAVELENGTH_MAX_A = float(wavelengths(_ENERGY_MIN_EV))
"""
Longest wavelength of the default range, in Angstroms.
"""


def process_compound(
    name: str,
    formula: str,
    spec: str,
    density: float,
    scaling_factor: float,
    temperature: float,
    wavelength_min: float = WAVELENGTH_MIN_A,
    wavelength_max: float = WAVELENGTH_MAX_A,
    points: int = _ENERGY_POINTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates AFGA plots for a given compound.

    Parameters
    ----------
    wavelength_min, wavelength_max : float
        Wavelength range to compute over, in angstroms. Defaults span the
        beamline's usable range.
    points : int
        Number of points across that range.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (wavelengths in angstroms, cross section in barns), ordered from the
        long-wavelength end down.

    Raises
    ------
    ValueError
        If the wavelength range is not positive and increasing.
    """
    if wavelength_min <= 0 or wavelength_max <= wavelength_min:
        raise ValueError(
            f"Wavelength range must be positive and increasing, got "
            f"{wavelength_min} to {wavelength_max} A."
        )
    # Built in the function's own unit; only NCrystal wants energies.
    wavelength_a = np.geomspace(wavelength_max, wavelength_min, points)
    energy_ev = energies(wavelength_a)

    composer = NC.NCMATComposer.from_hfg(
        spec, formula, density=density, title=name, debyetemp=temperature
    )
    material = composer.load()
    cross_section = material.scatter.crossSectionNonOriented(
        energy_ev
    ) + material.absorption.crossSectionNonOriented(energy_ev)
    cross_section = np.asarray(cross_section, dtype=float) * scaling_factor

    return wavelength_a, cross_section
