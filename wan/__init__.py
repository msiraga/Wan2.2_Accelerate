# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from . import configs, distributed, modules

# Core T2V imports (always available)
from .text2video import WanT2V

# Optional imports (require additional dependencies)
try:
    from .image2video import WanI2V
except ImportError:
    WanI2V = None

try:
    from .speech2video import WanS2V
except ImportError:
    WanS2V = None

try:
    from .textimage2video import WanTI2V
except ImportError:
    WanTI2V = None

try:
    from .animate import WanAnimate
except ImportError:
    WanAnimate = None