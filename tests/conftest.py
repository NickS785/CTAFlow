import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import CTAFlow as _cta_module
import CTAFlow.screeners as _cta_screeners

sys.modules.setdefault("CTAFlow.CTAFlow", _cta_module)
sys.modules.setdefault("CTAFlow.CTAFlow.screeners", _cta_screeners)
