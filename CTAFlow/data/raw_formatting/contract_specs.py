import pandas as pd
from typing import Dict, List, Union, Optional
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from copy import deepcopy


# ------------------------------
# Master mapping (15 liquid contracts).
# contract_unit: integer size only (no text)
# months: list[int] of delivery months 1..12
# min_tick: expressed in the same units as `unit`
# ------------------------------

CONTRACT_SPECS_RAW: Dict[str, dict] = {
    # --- Energy (NYMEX) ---
    "CL": {  # WTI Crude Oil
        "name": "WTI Crude Oil",
        "exchange": "NYMEX",
        "unit": "$/bbl",
        "contract_unit": 1000,
        "min_tick": 0.01,          # $/bbl
        "tick_value_usd": 10.00,
        "months": [1,2,3,4,5,6,7,8,9,10,11,12],
        "notes": "Physically delivered; light sweet crude."
    },
    "NG": {  # Henry Hub Natural Gas
        "name": "Henry Hub Natural Gas",
        "exchange": "NYMEX",
        "unit": "$/mmbtu",
        "contract_unit": 10000,
        "min_tick": 0.001,         # $/mmbtu
        "tick_value_usd": 10.00,
        "months": [1,2,3,4,5,6,7,8,9,10,11,12],
        "notes": "Physically delivered at Henry Hub."
    },
    "RB": {  # RBOB Gasoline
        "name": "RBOB Gasoline",
        "exchange": "NYMEX",
        "unit": "$/gal",
        "contract_unit": 42000,    # 1,000 bbl
        "min_tick": 0.0001,        # $/gal
        "tick_value_usd": 4.20,
        "months": [1,2,3,4,5,6,7,8,9,10,11,12],
        "notes": "NY Harbor delivery."
    },
    "HO": {  # NY Harbor ULSD
        "name": "NY Harbor ULSD",
        "exchange": "NYMEX",
        "unit": "$/gal",
        "contract_unit": 42000,
        "min_tick": 0.0001,        # $/gal
        "tick_value_usd": 4.20,
        "months": [1,2,3,4,5,6,7,8,9,10,11,12],
        "notes": "ULSD spec."
    },

    # --- Metals (COMEX/NYMEX) ---
    "GC": {  # Gold
        "name": "Gold",
        "exchange": "COMEX",
        "unit": "$/oz",
        "contract_unit": 100,
        "min_tick": 0.10,          # $/oz
        "tick_value_usd": 10.00,
        "months": [2,4,6,8,10,12],
        "notes": "995+ fineness."
    },
    "SI": {  # Silver
        "name": "Silver",
        "exchange": "COMEX",
        "unit": "$/oz",
        "contract_unit": 5000,
        "min_tick": 0.005,         # $/oz
        "tick_value_usd": 25.00,
        "months": [3,5,7,9,12],
        "notes": "Physically delivered."
    },
    "HG": {  # High Grade Copper
        "name": "High Grade Copper",
        "exchange": "COMEX",
        "unit": "cents/lb",        # express tick in cents to match unit
        "contract_unit": 25000,    # pounds
        "min_tick": 0.05,          # 0.05 cent/lb == $0.0005/lb
        "tick_value_usd": 12.50,   # 25,000 * $0.0005
        "months": [3,5,7,9,12],
        "notes": "Grade 1 cathode."
    },
    "PL": {  # Platinum
        "name": "Platinum",
        "exchange": "NYMEX",
        "unit": "$/oz",
        "contract_unit": 50,
        "min_tick": 0.10,          # $/oz
        "tick_value_usd": 5.00,
        "months": [1,4,7,10],
        "notes": "Physically delivered."
    },
    "PA": {  # Palladium
        "name": "Palladium",
        "exchange": "NYMEX",
        "unit": "$/oz",
        "contract_unit": 100,
        "min_tick": 0.10,          # $/oz
        "tick_value_usd": 10.00,
        "months": [3,6,9,12],
        "notes": "Physically delivered."
    },

    # --- Grains & Oilseeds (CBOT) ---
    "ZC": {  # Corn
        "name": "Corn",
        "exchange": "CBOT",
        "unit": "cents/bu",
        "contract_unit": 5000,
        "min_tick": 0.25,          # cents/bu
        "tick_value_usd": 12.50,
        "months": [3,5,7,9,12],
        "notes": "Bushel-based."
    },
    "ZS": {  # Soybeans
        "name": "Soybeans",
        "exchange": "CBOT",
        "unit": "cents/bu",
        "contract_unit": 5000,
        "min_tick": 0.25,          # cents/bu
        "tick_value_usd": 12.50,
        "months": [1,3,5,7,8,9,11],
        "notes": "Bushel-based."
    },
    "ZW": {  # Chicago SRW Wheat
        "name": "Chicago SRW Wheat",
        "exchange": "CBOT",
        "unit": "cents/bu",
        "contract_unit": 5000,
        "min_tick": 0.25,          # cents/bu
        "tick_value_usd": 12.50,
        "months": [3,5,7,9,12],
        "notes": "Bushel-based."
    },
    "ZM": {  # Soybean Meal
        "name": "Soybean Meal",
        "exchange": "CBOT",
        "unit": "$/ton",           # short ton
        "contract_unit": 100,      # short tons
        "min_tick": 0.10,          # $/ton
        "tick_value_usd": 10.00,
        "months": [1,3,5,7,8,9,10,12],
        "notes": "Short-ton (2,000 lb)."
    },
    "ZL": {  # Soybean Oil
        "name": "Soybean Oil",
        "exchange": "CBOT",
        "unit": "cents/lb",
        "contract_unit": 60000,    # pounds
        "min_tick": 0.01,          # 0.01 cent/lb == $0.0001/lb
        "tick_value_usd": 6.00,
        "months": [1,3,5,7,8,9,10,12],
        "notes": "Quoted in cents/lb."
    },

    # --- Livestock (CME) ---
    "LE": {  # Live Cattle
        "name": "Live Cattle",
        "exchange": "CME",
        "unit": "cents/lb",
        "contract_unit": 40000,    # pounds
        "min_tick": 0.025,         # 0.025 cent/lb == $0.00025/lb
        "tick_value_usd": 10.00,
        "months": [2,4,6,8,10,12],
        "notes": "Cash-settled."
    },
}


# ------------------------------
# Dataclass & loader
# ------------------------------

@dataclass(frozen=True)
class ContractSpecs:
    symbol: str
    name: str
    exchange: str
    unit: str                  # "$/bbl", "$/mmbtu", "$/gal", "$/oz", "cents/bu", "cents/lb", "$/ton"
    contract_unit: int         # e.g., 42000 (gal), 5000 (bu), 25000 (lb), 1000 (bbl)
    min_tick: float            # quoted in 'unit' terms (see comments above)
    tick_value_usd: float      # USD value per minimum tick
    months: List[int]          # delivery months as integers in 1..12
    notes: Optional[str] = None

    @staticmethod
    def _coerce_and_validate(sym: str, raw: dict) -> dict:
        """Normalize types and enforce invariants for one mapping."""
        spec = dict(raw)  # shallow copy (values are primitives)
        spec["name"] = str(spec["name"])
        spec["exchange"] = str(spec["exchange"])
        spec["unit"] = str(spec["unit"])
        spec["contract_unit"] = int(spec["contract_unit"])
        spec["min_tick"] = float(spec["min_tick"])
        spec["tick_value_usd"] = float(spec["tick_value_usd"])
        spec["months"] = [int(m) for m in spec["months"]]
        # Basic sanity checks
        if not all(1 <= m <= 12 for m in spec["months"]):
            raise ValueError(f"{sym}: months must be in 1..12, got {spec['months']}")
        if spec["contract_unit"] <= 0:
            raise ValueError(f"{sym}: contract_unit must be positive")
        return spec

    @classmethod
    def load_specs(cls, symbol: str, *overrides: dict) -> "ContractSpecs":
        """
        Load a single symbol's specs from CONTRACT_SPECS_RAW.
        Optionally pass one or more override dicts (merged left→right) to patch fields.

        Example:
            CL = ContractSpecs.load_specs("CL")
            CL_custom = ContractSpecs.load_specs("CL", {"min_tick": 0.02}, {"notes": "double tick"})
        """
        sym = symbol.upper()
        if sym not in CONTRACT_SPECS_RAW:
            available = ", ".join(sorted(CONTRACT_SPECS_RAW))
            raise KeyError(f"Unknown symbol '{symbol}'. Available: {available}")

        # Start with a deep copy of the raw record to avoid accidental mutation.
        merged = deepcopy(CONTRACT_SPECS_RAW[sym])

        # Apply any positional overrides in order.
        for patch in overrides:
            if patch:
                for k, v in patch.items():
                    merged[k] = v

        norm = cls._coerce_and_validate(sym, merged)
        return cls(
            symbol=sym,
            name=norm["name"],
            exchange=norm["exchange"],
            unit=norm["unit"],
            contract_unit=norm["contract_unit"],
            min_tick=norm["min_tick"],
            tick_value_usd=norm["tick_value_usd"],
            months=list(norm["months"]),
            notes=norm.get("notes"),
        )


@dataclass(frozen=True)
class ContractInfo:
    ticker: str
    month: str
    year: int
    exchange: str

    @property
    def contract_id(self) -> str:
        return f"{self.month}{str(self.year)[-2:]}"
