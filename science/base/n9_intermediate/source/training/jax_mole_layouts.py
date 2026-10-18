#!/usr/bin/env python3
"""Validated layout and domain definitions for farm-scale Mole cases."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


REFERENCE_DIAMETER_M = 126.0
REFERENCE_DOMAIN_LY_M = 500.0
REFERENCE_DX_M = 2394.0 / 192.0
REFERENCE_DZ_M = 882.0 / 72.0


@dataclass(frozen=True)
class MoleFarmLayout:
    """A fixed rectangular wind-farm layout and its LES domain."""

    name: str
    rows: int
    columns: int
    turbine_positions_m: tuple[tuple[float, float, float], ...]
    domain_lx_m: float
    domain_ly_m: float
    domain_lz_m: float
    nx: int
    ny: int
    nz: int
    diameter_m: float = REFERENCE_DIAMETER_M
    streamwise_spacing_d: float = 5.0
    spanwise_spacing_d: float = 5.0
    upstream_margin_d: float = 2.0
    downstream_margin_d: float = 7.0
    lateral_margin_d: float = 3.5

    @property
    def num_turbines(self) -> int:
        return len(self.turbine_positions_m)

    @property
    def dx_m(self) -> float:
        return self.domain_lx_m / (self.nx - 1)

    @property
    def dy_m(self) -> float:
        return self.domain_ly_m / (self.ny - 1)

    @property
    def dz_m(self) -> float:
        return self.domain_lz_m / self.nz

    @property
    def payload_sha256(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("layout name cannot be empty")
        if self.rows < 1 or self.columns < 1:
            raise ValueError("layout rows and columns must be positive")
        if self.num_turbines != self.rows * self.columns:
            raise ValueError(
                "turbine count does not match rows times columns"
            )
        if self.nx < 7 or self.ny < 7 or self.nz < 7:
            raise ValueError("LES grid is too small for sixth-order operators")
        if min(self.domain_lx_m, self.domain_ly_m, self.domain_lz_m) <= 0.0:
            raise ValueError("domain dimensions must be positive")

        half_diameter = 0.5 * self.diameter_m
        positions = self.turbine_positions_m
        for index, position in enumerate(positions):
            if len(position) != 3 or not all(math.isfinite(x) for x in position):
                raise ValueError(f"invalid turbine position at index {index}")
            x, y, z = position
            if not half_diameter <= x <= self.domain_lx_m - half_diameter:
                raise ValueError(f"turbine {index} rotor crosses x boundary")
            if not half_diameter <= y <= self.domain_ly_m - half_diameter:
                raise ValueError(f"turbine {index} rotor crosses y boundary")
            if not half_diameter <= z <= self.domain_lz_m - half_diameter:
                raise ValueError(f"turbine {index} rotor crosses z boundary")
            if x - 2.0 * self.diameter_m < -1.0e-9:
                raise ValueError(f"turbine {index} upstream probes cross inlet")
            if x + 3.0 * self.diameter_m > self.domain_lx_m + 1.0e-9:
                raise ValueError(f"turbine {index} downstream probes cross outlet")
            if z - self.diameter_m < -1.0e-9:
                raise ValueError(f"turbine {index} probes cross lower z boundary")
            if z + self.diameter_m > self.domain_lz_m + 1.0e-9:
                raise ValueError(f"turbine {index} probes cross upper z boundary")

        minimum_spacing = 2.0 * self.diameter_m
        for first in range(self.num_turbines):
            for second in range(first + 1, self.num_turbines):
                distance = math.dist(positions[first], positions[second])
                if distance < minimum_spacing:
                    raise ValueError(
                        f"turbines {first} and {second} are too close: {distance}"
                    )


def rectangular_mole_layout(
    rows: int,
    columns: int,
    *,
    diameter_m: float = REFERENCE_DIAMETER_M,
    hub_height_m: float = 90.0,
    streamwise_spacing_d: float = 5.0,
    spanwise_spacing_d: float = 5.0,
    upstream_margin_d: float = 2.0,
    downstream_margin_d: float = 7.0,
    lateral_margin_d: float = 3.5,
    domain_ly_m: float = REFERENCE_DOMAIN_LY_M,
) -> MoleFarmLayout:
    """Construct the canonical Mole-aligned rectangular layout family."""

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    if diameter_m <= 0.0:
        raise ValueError("diameter must be positive")
    if streamwise_spacing_d < 2.0 or spanwise_spacing_d < 2.0:
        raise ValueError("turbine spacing must be at least 2D")

    domain_lx_m = diameter_m * (
        upstream_margin_d
        + streamwise_spacing_d * (rows - 1)
        + downstream_margin_d
    )
    domain_lz_m = diameter_m * (
        2.0 * lateral_margin_d + spanwise_spacing_d * (columns - 1)
    )
    positions = tuple(
        (
            diameter_m * (upstream_margin_d + row * streamwise_spacing_d),
            hub_height_m,
            diameter_m * (lateral_margin_d + column * spanwise_spacing_d),
        )
        for row in range(rows)
        for column in range(columns)
    )
    nx = int(round(domain_lx_m / REFERENCE_DX_M)) + 1
    ny = int(round(domain_ly_m / (REFERENCE_DOMAIN_LY_M / 40.0))) + 1
    nz = int(round(domain_lz_m / REFERENCE_DZ_M))
    layout = MoleFarmLayout(
        name=f"mole_rect_{rows}x{columns}",
        rows=rows,
        columns=columns,
        turbine_positions_m=positions,
        domain_lx_m=domain_lx_m,
        domain_ly_m=domain_ly_m,
        domain_lz_m=domain_lz_m,
        nx=nx,
        ny=ny,
        nz=nz,
        diameter_m=diameter_m,
        streamwise_spacing_d=streamwise_spacing_d,
        spanwise_spacing_d=spanwise_spacing_d,
        upstream_margin_d=upstream_margin_d,
        downstream_margin_d=downstream_margin_d,
        lateral_margin_d=lateral_margin_d,
    )
    layout.validate()
    return layout


def canonical_scale_layouts() -> dict[int, MoleFarmLayout]:
    layouts = {
        3: rectangular_mole_layout(3, 1),
        9: rectangular_mole_layout(3, 3),
        25: rectangular_mole_layout(5, 5),
        50: rectangular_mole_layout(5, 10),
        81: rectangular_mole_layout(9, 9),
        100: rectangular_mole_layout(10, 10),
    }
    if set(layouts) != {layout.num_turbines for layout in layouts.values()}:
        raise AssertionError("canonical layout turbine-count mismatch")
    return layouts
