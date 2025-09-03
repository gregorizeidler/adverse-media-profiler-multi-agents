from dataclasses import dataclass
from typing import List, Optional

@dataclass
class APIConfig:
    openai_api_key: str
    news_api_key: str
    serper_api_key: str

@dataclass
class SearchConfig:
    max_articles: int = 50
    max_age_days: int = 90
    risk_categories: List[str] = None

    def __post_init__(self):
        if self.risk_categories is None:
            self.risk_categories = [
                "financial_crime",
                "regulatory_violation",
                "legal_issues",
                "reputational_damage",
                "operational_risk",
                "fraud",
                "corruption",
                "money_laundering"
            ]

@dataclass
class Config:
    api: APIConfig
    search: SearchConfig
