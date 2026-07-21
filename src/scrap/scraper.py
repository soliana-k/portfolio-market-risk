import numpy as np
import pandas as pd
from requests_ratelimiter import LimitedSession
from typing import List, Optional, Dict

class CollectPortfolio:
    def __init__(self, portfolio: Dict):
        self.portfolio=portfolio
        self.rate_limiter=LimitedSession(per_second=1)
