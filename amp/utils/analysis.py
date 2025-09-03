from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RiskAssessment:
    risk_level: str  # HIGH, MEDIUM, LOW
    risk_type: str
    description: str
    source_credibility: str
    timestamp: datetime

@dataclass
class MediaMention:
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    risk_assessment: RiskAssessment = None

class AnalysisUtils:
    # Source credibility database
    CREDIBLE_SOURCES = {
        # Tier 1: Highly credible (0.9-1.0)
        'reuters': 0.95, 'bloomberg': 0.94, 'financial times': 0.93,
        'wall street journal': 0.92, 'associated press': 0.91,
        
        # Tier 2: Generally reliable (0.7-0.89)
        'bbc': 0.85, 'cnn': 0.78, 'nbc': 0.79, 'abc': 0.80,
        'the guardian': 0.82, 'washington post': 0.83,
        
        # Tier 3: Moderate reliability (0.5-0.69)
        'fox news': 0.65, 'huffpost': 0.60, 'buzzfeed news': 0.58,
        
        # Tier 4: Low reliability (0.3-0.49)
        'social media': 0.35, 'blog': 0.40, 'forum': 0.30
    }
    
    # Risk categories and associated keywords
    RISK_CATEGORIES = {
        'financial_crime': [
            'fraud', 'embezzlement', 'money laundering', 'tax evasion',
            'insider trading', 'bribery', 'corruption', 'ponzi scheme'
        ],
        'legal_issues': [
            'lawsuit', 'litigation', 'conviction', 'indictment', 'arrest',
            'investigation', 'criminal charges', 'court case', 'trial'
        ],
        'regulatory_violation': [
            'violation', 'non-compliance', 'regulatory action', 'fine',
            'penalty', 'sanctions', 'license revoked', 'cease and desist'
        ],
        'reputational_damage': [
            'scandal', 'controversy', 'misconduct', 'unethical',
            'discrimination', 'harassment', 'cover-up', 'whistleblower'
        ],
        'operational_risk': [
            'data breach', 'cyber attack', 'system failure', 'outage',
            'security incident', 'operational failure', 'service disruption'
        ],
        'financial_distress': [
            'bankruptcy', 'insolvency', 'financial difficulty', 'debt default',
            'liquidity crisis', 'restructuring', 'bailout', 'receivership'
        ]
    }

    @staticmethod
    def assess_source_credibility(source: str) -> str:
        """
        Assess the credibility of an information source.
        
        Args:
            source: Name or URL of the source
            
        Returns:
            str: Credibility level ('HIGH', 'MEDIUM', 'LOW', 'UNKNOWN')
        """
        source_lower = source.lower()
        
        # Check against known sources
        for known_source, score in AnalysisUtils.CREDIBLE_SOURCES.items():
            if known_source in source_lower:
                if score >= 0.9:
                    return 'HIGH'
                elif score >= 0.7:
                    return 'MEDIUM'
                else:
                    return 'LOW'
        
        # Check for common indicators of credibility
        credibility_indicators = {
            'government': 0.85,
            'official': 0.80,
            'university': 0.85,
            'research': 0.80,
            'academic': 0.85
        }
        
        for indicator, score in credibility_indicators.items():
            if indicator in source_lower:
                return 'HIGH' if score >= 0.8 else 'MEDIUM'
        
        # Check for low-credibility indicators
        low_credibility = ['anonymous', 'rumor', 'unverified', 'alleged']
        if any(indicator in source_lower for indicator in low_credibility):
            return 'LOW'
        
        return 'UNKNOWN'

    @staticmethod
    def categorize_risk(text: str) -> List[str]:
        """
        Categorize risks found in text using NLP and keyword analysis.
        
        Args:
            text: Text to analyze for risk categories
            
        Returns:
            List[str]: List of identified risk categories
        """
        text_lower = text.lower()
        identified_risks = []
        
        # Check each risk category
        for category, keywords in AnalysisUtils.RISK_CATEGORIES.items():
            matches = 0
            for keyword in keywords:
                if keyword in text_lower:
                    matches += 1
            
            # Add category if significant keyword presence (threshold: 10% of keywords)
            if matches >= max(1, len(keywords) * 0.1):
                identified_risks.append({
                    'category': category,
                    'confidence': min(1.0, matches / len(keywords)),
                    'matched_keywords': [kw for kw in keywords if kw in text_lower]
                })
        
        # Sort by confidence and return category names
        identified_risks.sort(key=lambda x: x['confidence'], reverse=True)
        return [risk['category'] for risk in identified_risks]

    @staticmethod
    def calculate_risk_score(mentions: List[MediaMention]) -> float:
        """
        Calculate overall risk score based on media mentions.
        
        Args:
            mentions: List of MediaMention objects
            
        Returns:
            float: Overall risk score (0.0 to 1.0)
        """
        if not mentions:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for mention in mentions:
            # Get source credibility weight
            credibility = AnalysisUtils.assess_source_credibility(mention.source)
            
            credibility_weights = {
                'HIGH': 1.0,
                'MEDIUM': 0.7,
                'LOW': 0.4,
                'UNKNOWN': 0.5
            }
            
            source_weight = credibility_weights.get(credibility, 0.5)
            
            # Analyze risk categories in content
            risk_categories = AnalysisUtils.categorize_risk(mention.content)
            
            # Calculate content risk score
            content_risk = 0.0
            if risk_categories:
                # Weight different risk types
                risk_weights = {
                    'financial_crime': 0.9,
                    'legal_issues': 0.8,
                    'regulatory_violation': 0.7,
                    'reputational_damage': 0.6,
                    'operational_risk': 0.5,
                    'financial_distress': 0.7
                }
                
                category_scores = [risk_weights.get(cat, 0.5) for cat in risk_categories]
                content_risk = min(1.0, sum(category_scores) / len(risk_categories))
            
            # Time decay factor (recent mentions are more important)
            if mention.published_date:
                days_old = (datetime.now() - mention.published_date).days
                time_factor = max(0.1, 1.0 - (days_old / 365))  # Decay over a year
            else:
                time_factor = 0.5  # Default for unknown dates
            
            # Calculate weighted score for this mention
            mention_score = content_risk * source_weight * time_factor
            
            total_weighted_score += mention_score * source_weight
            total_weight += source_weight
        
        # Calculate final risk score
        if total_weight > 0:
            base_score = total_weighted_score / total_weight
            
            # Frequency boost (more mentions = higher risk)
            frequency_factor = min(2.0, 1.0 + (len(mentions) - 1) * 0.1)
            
            final_score = min(1.0, base_score * frequency_factor)
        else:
            final_score = 0.0
        
        return round(final_score, 3)
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using advanced NLP.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities with metadata
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except:
            # Fallback to simple extraction if spaCy not available
            return AnalysisUtils._simple_entity_extraction(text)
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # spaCy doesn't provide confidence scores directly
            })
        
        return entities
    
    @staticmethod
    def _simple_entity_extraction(text: str) -> List[Dict[str, Any]]:
        """
        Fallback entity extraction using regex patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of simple entities
        """
        import re
        
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'MONEY': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',
            'PERCENT': r'\d+\.?\d*%',
            'DATE': r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}-\d{1,2}-\d{2,4}\b'
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return entities
    
    @staticmethod
    def sentiment_analysis(text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity,  # 0 (objective) to 1 (subjective)
                'sentiment_label': AnalysisUtils._get_sentiment_label(blob.sentiment.polarity)
            }
        except ImportError:
            # Simple keyword-based fallback
            return AnalysisUtils._simple_sentiment_analysis(text)
    
    @staticmethod
    def _get_sentiment_label(polarity: float) -> str:
        """Convert polarity score to label."""
        if polarity > 0.1:
            return 'POSITIVE'
        elif polarity < -0.1:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    @staticmethod
    def _simple_sentiment_analysis(text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using keyword lists."""
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'success', 'win', 'growth',
            'profit', 'expansion', 'achievement', 'award', 'innovation'
        ]
        
        negative_words = [
            'bad', 'terrible', 'negative', 'loss', 'failure', 'decline', 'scandal',
            'fraud', 'corruption', 'violation', 'penalty', 'lawsuit', 'crisis'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment_label': 'NEUTRAL'}
        
        # Simple polarity calculation
        polarity = (positive_count - negative_count) / max(total_words, 1)
        subjectivity = (positive_count + negative_count) / max(total_words, 1)
        
        return {
            'polarity': max(-1.0, min(1.0, polarity)),
            'subjectivity': min(1.0, subjectivity),
            'sentiment_label': AnalysisUtils._get_sentiment_label(polarity)
        }
