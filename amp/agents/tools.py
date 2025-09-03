from typing import Dict, List, Optional, Any
import os
import json
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta
import networkx as nx
import spacy
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from textblob import TextBlob
from crewai_tools import tool

class LocalProcessingTools:
    # Carregar o modelo spaCy para processamento de linguagem natural
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def analyze_local_news(target: str, data_dir: str = "data/news") -> List[Dict]:
        """
        Analyzes local news files for relevant information about the target entity.
        
        Args:
            target: Name of the target entity
            data_dir: Directory containing news files
        """
        results = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            data_path.mkdir(parents=True)
            return []

        for file in data_path.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                doc = LocalProcessingTools.nlp(content)
                
                # Check if the target is mentioned in the text
                if re.search(rf"\b{target}\b", content, re.IGNORECASE):
                    # Extract related named entities
                    entities = [(ent.text, ent.label_) for ent in doc.ents]
                    
                    # Extract sentences that mention the target
                    relevant_sentences = []
                    for sent in doc.sents:
                        if re.search(rf"\b{target}\b", sent.text, re.IGNORECASE):
                            relevant_sentences.append(sent.text)
                    
                    results.append({
                        "source": file.name,
                        "content": content,
                        "relevant_sentences": relevant_sentences,
                        "entities": entities
                    })
        
        return results

    @staticmethod
    def process_documents(target: str, data_dir: str = "data/documents") -> List[Dict]:
        """
        Processes local documents (PDFs, DOCs, etc.) converted to text format.
        
        Args:
            target: Name of the target entity
            data_dir: Directory containing the documents
        """
        results = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            data_path.mkdir(parents=True)
            return []

        for file in data_path.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
                # Similar ao analyze_local_news, mas com foco em documentos
                doc = LocalProcessingTools.nlp(content)
                
                if re.search(rf"\b{target}\b", content, re.IGNORECASE):
                    results.append({
                        "source": file.name,
                        "content": content,
                        "summary": LocalProcessingTools._generate_summary(doc)
                    })
        
        return results

    @staticmethod
    def check_local_databases(target: str, db_dir: str = "data/databases") -> Dict:
        """
        Checks local databases (sanctions lists, PEPs, etc.).
        
        Args:
            target: Name of the target entity
            db_dir: Directory containing the databases
        """
        results = {
            "sanctions": [],
            "pep": [],
            "corporate": []
        }
        
        db_path = Path(db_dir)
        if not db_path.exists():
            db_path.mkdir(parents=True)
            return results

        # Verificar cada tipo de base de dados
        for db_type in ["sanctions", "pep", "corporate"]:
            db_file = db_path / f"{db_type}.json"
            if db_file.exists():
                with open(db_file, "r", encoding="utf-8") as f:
                    database = json.load(f)
                    matches = [entry for entry in database 
                             if target.lower() in entry.get("name", "").lower()]
                    results[db_type].extend(matches)
        
        return results

    @staticmethod
    def entity_disambiguation(mentions: List[Dict], target: Dict) -> List[Dict]:
        """
        Disambiguates entity mentions to confirm they refer to the target.
        
        Args:
            mentions: List of found mentions
            target: Target information for comparison
        """
        verified_mentions = []
        
        for mention in mentions:
            confidence = LocalProcessingTools._calculate_similarity(
                mention.get("text", ""),
                target.get("name", "")
            )
            
            if confidence > 0.8:  # Confidence threshold
                verified_mentions.append({
                    **mention,
                    "confidence": confidence
                })
        
        return verified_mentions

    @staticmethod
    def analyze_financial_risk(data: List[Dict]) -> Dict:
        """
        Analyzes financial risks from collected data.
        """
        risk_patterns = {
            "fraud": r"\b(fraud|embezzlement|scheme)\b",
            "money_laundering": r"\b(money laundering|illicit funds|suspicious transactions)\b",
            "bankruptcy": r"\b(bankruptcy|insolvency|financial difficulty)\b"
        }
        
        risks = {category: [] for category in risk_patterns}
        
        for item in data:
            content = item.get("content", "")
            for category, pattern in risk_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Extrair contexto ao redor do match
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    risks[category].append({
                        "source": item.get("source", "unknown"),
                        "context": context,
                        "date": item.get("date", "unknown")
                    })
        
        return risks

    @staticmethod
    def analyze_legal_risk(data: List[Dict]) -> Dict:
        """
        Analyzes legal and criminal risks.
        """
        # Similar to analyze_financial_risk, but with different patterns
        risk_patterns = {
            "criminal": r"\b(arrest|convicted|indicted|investigation)\b",
            "regulatory": r"\b(violation|non-compliance|regulatory action)\b",
            "litigation": r"\b(lawsuit|legal action|court)\b"
        }
        
        risks = {category: [] for category in risk_patterns}
        
        for item in data:
            content = item.get("content", "")
            for category, pattern in risk_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    risks[category].append({
                        "source": item.get("source", "unknown"),
                        "context": context,
                        "date": item.get("date", "unknown")
                    })
        
        return risks

    @staticmethod
    def analyze_reputational_risk(data: List[Dict]) -> Dict:
        """
        Analyzes reputational and ethical risks.
        """
        risk_patterns = {
            "ethical": r"\b(ethics|misconduct|scandal)\b",
            "environmental": r"\b(pollution|environmental damage|climate)\b",
            "social": r"\b(discrimination|labor|working conditions)\b",
            "governance": r"\b(corruption|bribery|conflict of interest)\b"
        }
        
        risks = {category: [] for category in risk_patterns}
        
        for item in data:
            content = item.get("content", "")
            for category, pattern in risk_patterns.items():
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 100)
                    end = min(len(content), match.end() + 100)
                    context = content[start:end]
                    
                    risks[category].append({
                        "source": item.get("source", "unknown"),
                        "context": context,
                        "date": item.get("date", "unknown")
                    })
        
        return risks

    @staticmethod
    def analyze_connections(data: List[Dict]) -> Dict:
        """
        Analyzes and maps connections between entities.
        """
        G = nx.Graph()
        connections = []
        
        for item in data:
            content = item.get("content", "")
            doc = LocalProcessingTools.nlp(content)
            
            # Extract entities and their relationships
            entities = list(doc.ents)
            
            for i, ent1 in enumerate(entities):
                if ent1.label_ in ["PERSON", "ORG"]:
                    for ent2 in entities[i+1:]:
                        if ent2.label_ in ["PERSON", "ORG"]:
                            # Find the sentence containing both entities
                            for sent in doc.sents:
                                if ent1.text in sent.text and ent2.text in sent.text:
                                    connection = {
                                        "entity1": ent1.text,
                                        "entity2": ent2.text,
                                        "context": sent.text,
                                        "source": item.get("source", "unknown")
                                    }
                                    connections.append(connection)
                                    
                                    # Adicionar ao grafo
                                    G.add_edge(ent1.text, ent2.text)
        
        return {
            "connections": connections,
            "graph": nx.node_link_data(G)  # Converter grafo para formato JSON
        }

    @staticmethod
    def verify_findings(findings: Dict) -> Dict:
        """
        Verifies and validates findings.
        """
        verified = {
            "confirmed": [],
            "suspicious": [],
            "rejected": []
        }
        
        for category, items in findings.items():
            for item in items:
                # Check if there are multiple sources confirming the information
                sources = set([i.get("source") for i in items 
                             if i.get("context") == item.get("context")])
                
                if len(sources) >= 2:
                    verified["confirmed"].append({
                        **item,
                        "verification": {
                            "status": "confirmed",
                            "sources": list(sources)
                        }
                    })
                elif len(sources) == 1:
                    verified["suspicious"].append({
                        **item,
                        "verification": {
                            "status": "needs_verification",
                            "sources": list(sources)
                        }
                    })
                else:
                    verified["rejected"].append({
                        **item,
                        "verification": {
                            "status": "insufficient_evidence",
                            "sources": list(sources)
                        }
                    })
        
        return verified

    @staticmethod
    def compile_report(data: Dict) -> Dict:
        """
        Compiles the final report with all findings.
        """
        now = datetime.now()
        
        report = {
            "metadata": {
                "generated_at": now.isoformat(),
                "target": data.get("target", {}),
                "version": "1.0"
            },
            "executive_summary": LocalProcessingTools._generate_executive_summary(data),
            "risk_assessment": {
                "financial_risks": data.get("financial_risks", {}),
                "legal_risks": data.get("legal_risks", {}),
                "reputational_risks": data.get("reputational_risks", {})
            },
            "network_analysis": data.get("connections", {}),
            "verification_status": data.get("verification", {}),
            "recommendations": LocalProcessingTools._generate_recommendations(data)
        }
        
        return report

    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculates similarity between two texts using spaCy.
        """
        doc1 = LocalProcessingTools.nlp(text1)
        doc2 = LocalProcessingTools.nlp(text2)
        return doc1.similarity(doc2)

    @staticmethod
    def _generate_summary(doc) -> str:
        """
        Generates a document summary using the most important sentences.
        """
        # Simplified implementation - takes the first 3 sentences
        return " ".join([sent.text for sent in list(doc.sents)[:3]])

    @staticmethod
    def _generate_executive_summary(data: Dict) -> str:
        """
        Generates an executive summary of findings.
        """
        summary_parts = []
        
        # Add main target information
        target = data.get("target", {})
        if target:
            summary_parts.append(f"Target Entity: {target.get('name', 'Unknown')}")
        
        # Resumir riscos encontrados
        risk_types = ["financial_risks", "legal_risks", "reputational_risks"]
        for risk_type in risk_types:
            risks = data.get(risk_type, {})
            if risks:
                count = sum(len(items) for items in risks.values())
                summary_parts.append(f"Found {count} {risk_type.replace('_', ' ')}.")
        
        # Add connection information
        connections = data.get("connections", {})
        if connections:
            network_size = len(connections.get("graph", {}).get("nodes", []))
            summary_parts.append(f"Identified network of {network_size} connected entities.")
        
        return "\n".join(summary_parts)

    @staticmethod
    def _generate_recommendations(data: Dict) -> List[str]:
        """
        Generates recommendations based on findings.
        """
        recommendations = []
        
        # Analisar riscos financeiros
        financial_risks = data.get("financial_risks", {})
        if financial_risks.get("fraud") or financial_risks.get("money_laundering"):
            recommendations.append(
                "Conduct detailed financial audit and enhance transaction monitoring."
            )
        
        # Analisar riscos legais
        legal_risks = data.get("legal_risks", {})
        if legal_risks.get("regulatory") or legal_risks.get("litigation"):
            recommendations.append(
                "Review compliance procedures and seek legal counsel."
            )
        
        # Analisar riscos reputacionais
        reputational_risks = data.get("reputational_risks", {})
        if reputational_risks.get("ethical") or reputational_risks.get("social"):
            recommendations.append(
                "Develop crisis communication plan and enhance ESG policies."
            )
        
        return recommendations


class ExternalAPITools:
    """Tools for integrating with external APIs and services."""
    
    @staticmethod
    @tool("Search news via external API")
    def search_news_api(query: str, days_back: int = 30) -> List[Dict]:
        """
        Search for news articles using external news APIs.
        
        Args:
            query: Search query
            days_back: Number of days to look back
        """
        # Simulated news API integration - replace with real API calls
        articles = []
        
        # Mock data for demonstration
        mock_articles = [
            {
                "title": f"Financial investigation regarding {query}",
                "content": f"Recent developments show {query} under scrutiny for financial irregularities...",
                "source": "Financial Times",
                "url": "https://example.com/article1",
                "published_date": datetime.now() - timedelta(days=5),
                "relevance_score": 0.95
            },
            {
                "title": f"Legal proceedings involving {query}",
                "content": f"Court documents reveal {query} facing multiple legal challenges...",
                "source": "Reuters",
                "url": "https://example.com/article2", 
                "published_date": datetime.now() - timedelta(days=12),
                "relevance_score": 0.87
            }
        ]
        
        articles.extend(mock_articles)
        
        return articles

    @staticmethod
    @tool("Search social media mentions")
    def search_social_media(target: str, platforms: List[str] = None) -> List[Dict]:
        """
        Search social media platforms for mentions of the target.
        
        Args:
            target: Target entity to search for
            platforms: List of platforms to search (default: all available)
        """
        if platforms is None:
            platforms = ['twitter', 'linkedin', 'reddit']
        
        mentions = []
        
        # Mock social media data
        mock_mentions = [
            {
                "platform": "twitter",
                "content": f"Concerns raised about {target}'s business practices",
                "author": "@whistleblower123",
                "engagement": {"likes": 45, "retweets": 12, "replies": 8},
                "timestamp": datetime.now() - timedelta(hours=6),
                "sentiment": "negative"
            },
            {
                "platform": "linkedin",
                "content": f"Congratulations to {target} on their recent expansion",
                "author": "Industry Insider",
                "engagement": {"likes": 23, "comments": 5, "shares": 2},
                "timestamp": datetime.now() - timedelta(days=3),
                "sentiment": "positive"
            }
        ]
        
        mentions.extend(mock_mentions)
        
        return mentions

    @staticmethod
    @tool("Query public records")
    def query_public_records(target: str, record_types: List[str] = None) -> List[Dict]:
        """
        Query public record databases for information about the target.
        
        Args:
            target: Target entity
            record_types: Types of records to search
        """
        if record_types is None:
            record_types = ['corporate', 'legal', 'property', 'regulatory']
        
        records = []
        
        # Mock public records
        mock_records = [
            {
                "record_type": "corporate",
                "source": "Corporate Registry",
                "data": {
                    "company_name": target,
                    "status": "Active",
                    "incorporation_date": "2010-03-15",
                    "directors": ["John Smith", "Jane Doe"],
                    "address": "123 Business St, City, State"
                },
                "last_updated": datetime.now() - timedelta(days=30)
            },
            {
                "record_type": "regulatory",
                "source": "Financial Regulatory Authority",
                "data": {
                    "subject": target,
                    "violations": [
                        {"date": "2023-06-15", "type": "Minor reporting violation", "fine": "$5,000"}
                    ]
                },
                "last_updated": datetime.now() - timedelta(days=90)
            }
        ]
        
        records.extend(mock_records)
        
        return records


class AdvancedAnalysisTools:
    """Advanced analysis tools for sophisticated intelligence processing."""
    
    @staticmethod
    @tool("Perform entity disambiguation")
    def entity_disambiguation(mentions: List[Dict], target_info: Dict) -> List[Dict]:
        """
        Advanced entity disambiguation using multiple techniques.
        
        Args:
            mentions: List of entity mentions to disambiguate
            target_info: Known information about the target entity
        """
        disambiguated = []
        
        target_name = target_info.get('name', '').lower()
        target_aliases = target_info.get('aliases', [])
        
        for mention in mentions:
            mention_text = mention.get('content', '').lower()
            mention_title = mention.get('title', '').lower()
            
            # Calculate similarity scores
            name_similarity = AdvancedAnalysisTools._calculate_fuzzy_match(
                target_name, mention_text
            )
            
            # Context analysis
            context_score = AdvancedAnalysisTools._analyze_context_relevance(
                mention_text, target_info
            )
            
            # Combined confidence score
            confidence = (name_similarity * 0.4 + context_score * 0.6)
            
            if confidence > 0.7:  # Threshold for inclusion
                disambiguated.append({
                    **mention,
                    'disambiguation': {
                        'confidence': confidence,
                        'name_similarity': name_similarity,
                        'context_relevance': context_score,
                        'verified': confidence > 0.85
                    }
                })
        
        return disambiguated

    @staticmethod
    @tool("Verify source credibility")
    def source_verification(sources: List[str]) -> Dict[str, Dict]:
        """
        Assess the credibility and reliability of information sources.
        
        Args:
            sources: List of source names/URLs to verify
        """
        credibility_db = {
            # Tier 1: Highly credible
            'reuters': {'score': 0.95, 'tier': 1, 'bias': 'neutral'},
            'financial times': {'score': 0.94, 'tier': 1, 'bias': 'slight_center'},
            'bloomberg': {'score': 0.93, 'tier': 1, 'bias': 'neutral'},
            'wall street journal': {'score': 0.92, 'tier': 1, 'bias': 'slight_center'},
            
            # Tier 2: Generally reliable
            'cnn': {'score': 0.78, 'tier': 2, 'bias': 'left_center'},
            'bbc': {'score': 0.85, 'tier': 2, 'bias': 'neutral'},
            'the guardian': {'score': 0.82, 'tier': 2, 'bias': 'left_center'},
            
            # Tier 3: Moderate reliability
            'social media': {'score': 0.35, 'tier': 3, 'bias': 'varies'},
            'anonymous blog': {'score': 0.25, 'tier': 3, 'bias': 'unknown'}
        }
        
        source_assessments = {}
        
        for source in sources:
            source_lower = source.lower()
            
            # Check against known sources
            assessment = None
            for known_source, data in credibility_db.items():
                if known_source in source_lower:
                    assessment = data.copy()
                    break
            
            if assessment is None:
                # Default assessment for unknown sources
                assessment = {
                    'score': 0.5,
                    'tier': 2,
                    'bias': 'unknown',
                    'note': 'Unknown source - manual verification recommended'
                }
            
            source_assessments[source] = assessment
        
        return source_assessments

    @staticmethod
    @tool("Calculate confidence scores")
    def confidence_scoring(data: Dict) -> Dict:
        """
        Calculate comprehensive confidence scores for findings.
        
        Args:
            data: Data to score including sources, mentions, etc.
        """
        scores = {
            'overall_confidence': 0.0,
            'source_reliability': 0.0,
            'information_consistency': 0.0,
            'temporal_relevance': 0.0,
            'corroboration_level': 0.0
        }
        
        sources = data.get('sources', [])
        mentions = data.get('mentions', [])
        
        if sources:
            source_scores = [s.get('credibility', 0.5) for s in sources]
            scores['source_reliability'] = np.mean(source_scores)
        
        if mentions:
            # Check for consistency across mentions
            sentiment_scores = [m.get('sentiment_score', 0.0) for m in mentions]
            consistency = 1 - np.std(sentiment_scores) if len(sentiment_scores) > 1 else 1.0
            scores['information_consistency'] = max(0, consistency)
            
            # Temporal relevance
            now = datetime.now()
            dates = [m.get('date', now) for m in mentions]
            if isinstance(dates[0], str):
                dates = [datetime.fromisoformat(d) if d else now for d in dates]
            
            days_old = [(now - d).days for d in dates]
            avg_age = np.mean(days_old)
            scores['temporal_relevance'] = max(0, 1 - (avg_age / 365))  # Decay over year
            
            # Corroboration level
            scores['corroboration_level'] = min(1.0, len(mentions) / 3)  # More mentions = better
        
        # Overall confidence is weighted average
        weights = [0.3, 0.25, 0.2, 0.25]  # source, consistency, temporal, corroboration
        values = [scores['source_reliability'], scores['information_consistency'], 
                 scores['temporal_relevance'], scores['corroboration_level']]
        
        scores['overall_confidence'] = np.average(values, weights=weights)
        
        return scores

    @staticmethod
    @tool("Perform comprehensive financial risk analysis")
    def financial_risk_analysis(data: List[Dict]) -> Dict:
        """
        Advanced financial risk analysis using multiple indicators.
        
        Args:
            data: Financial data and mentions for analysis
        """
        risk_indicators = {
            'fraud_risk': {'score': 0.0, 'indicators': []},
            'money_laundering_risk': {'score': 0.0, 'indicators': []},
            'regulatory_risk': {'score': 0.0, 'indicators': []},
            'operational_risk': {'score': 0.0, 'indicators': []},
            'reputational_risk': {'score': 0.0, 'indicators': []}
        }
        
        # Advanced pattern matching
        fraud_patterns = [
            r'\b(?:fraud|embezzl|scheme|ponzi|pyramid)\w*\b',
            r'\b(?:misappropriat|steal|theft)\w*\b',
            r'\b(?:fake|false|fictitious)\s+(?:document|record|statement)\b'
        ]
        
        ml_patterns = [
            r'\b(?:money\s+launder|suspicious\s+transaction|cash\s+structur)\w*\b',
            r'\b(?:shell\s+compan|front\s+business|nominee)\w*\b',
            r'\b(?:bulk\s+cash|currency\s+exchange|hawala)\b'
        ]
        
        for item in data:
            content = item.get('content', '').lower()
            
            # Fraud risk analysis
            fraud_matches = []
            for pattern in fraud_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                fraud_matches.extend(matches)
            
            if fraud_matches:
                risk_indicators['fraud_risk']['score'] += len(fraud_matches) * 0.2
                risk_indicators['fraud_risk']['indicators'].extend(fraud_matches)
            
            # Money laundering risk
            ml_matches = []
            for pattern in ml_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                ml_matches.extend(matches)
            
            if ml_matches:
                risk_indicators['money_laundering_risk']['score'] += len(ml_matches) * 0.25
                risk_indicators['money_laundering_risk']['indicators'].extend(ml_matches)
        
        # Normalize scores
        for risk_type in risk_indicators:
            risk_indicators[risk_type]['score'] = min(1.0, risk_indicators[risk_type]['score'])
        
        return risk_indicators

    @staticmethod
    @tool("Perform legal risk analysis")
    def legal_risk_analysis(data: List[Dict]) -> Dict:
        """
        Comprehensive legal risk assessment.
        
        Args:
            data: Legal-related data for analysis
        """
        legal_risks = {
            'criminal_risk': {'score': 0.0, 'cases': []},
            'civil_litigation_risk': {'score': 0.0, 'cases': []},
            'regulatory_risk': {'score': 0.0, 'violations': []},
            'compliance_risk': {'score': 0.0, 'issues': []}
        }
        
        criminal_patterns = [
            r'\b(?:arrest|convicted|indicted|charged|sentenced)\b',
            r'\b(?:criminal|felony|misdemeanor)\b',
            r'\b(?:investigation|probe|inquiry)\b'
        ]
        
        for item in data:
            content = item.get('content', '').lower()
            
            # Criminal risk assessment
            for pattern in criminal_patterns:
                if re.search(pattern, content):
                    legal_risks['criminal_risk']['score'] += 0.3
                    legal_risks['criminal_risk']['cases'].append({
                        'source': item.get('source'),
                        'summary': content[:200] + '...'
                    })
        
        # Normalize scores
        for risk_type in legal_risks:
            legal_risks[risk_type]['score'] = min(1.0, legal_risks[risk_type]['score'])
        
        return legal_risks

    @staticmethod
    @tool("Perform reputational risk analysis")
    def reputational_risk_analysis(data: List[Dict]) -> Dict:
        """
        Advanced reputational risk analysis including sentiment analysis.
        
        Args:
            data: Reputational data for analysis
        """
        reputation_analysis = {
            'overall_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'key_themes': [],
            'risk_score': 0.0,
            'recommendations': []
        }
        
        sentiments = []
        
        for item in data:
            content = item.get('content', '')
            
            # Sentiment analysis using TextBlob
            blob = TextBlob(content)
            sentiment_score = blob.sentiment.polarity
            sentiments.append(sentiment_score)
            
            # Categorize sentiment
            if sentiment_score > 0.1:
                reputation_analysis['sentiment_distribution']['positive'] += 1
            elif sentiment_score < -0.1:
                reputation_analysis['sentiment_distribution']['negative'] += 1
            else:
                reputation_analysis['sentiment_distribution']['neutral'] += 1
        
        if sentiments:
            reputation_analysis['overall_sentiment'] = np.mean(sentiments)
            reputation_analysis['risk_score'] = max(0, -reputation_analysis['overall_sentiment'])
        
        return reputation_analysis

    @staticmethod
    @tool("Perform network analysis")
    def network_analysis(entities: List[Dict]) -> Dict:
        """
        Advanced network analysis to identify key relationships and influence.
        
        Args:
            entities: List of entities and their connections
        """
        G = nx.Graph()
        
        # Build network
        for entity in entities:
            entity_name = entity.get('name')
            connections = entity.get('connections', [])
            
            G.add_node(entity_name, **entity)
            
            for connection in connections:
                G.add_edge(entity_name, connection.get('name'), 
                          weight=connection.get('strength', 1))
        
        # Network analysis
        analysis = {
            'network_size': len(G.nodes()),
            'connection_count': len(G.edges()),
            'density': nx.density(G),
            'key_players': {},
            'communities': [],
            'risk_paths': []
        }
        
        if len(G.nodes()) > 0:
            # Calculate centrality measures
            centrality_measures = {
                'betweenness': nx.betweenness_centrality(G),
                'degree': nx.degree_centrality(G),
                'closeness': nx.closeness_centrality(G)
            }
            
            # Identify key players
            for measure_name, centrality in centrality_measures.items():
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                analysis['key_players'][measure_name] = top_nodes
        
        return analysis

    @staticmethod
    @tool("Perform sentiment analysis")
    def sentiment_analysis(texts: List[str]) -> Dict:
        """
        Advanced sentiment analysis across multiple texts.
        
        Args:
            texts: List of texts to analyze
        """
        results = {
            'overall_sentiment': 0.0,
            'sentiment_scores': [],
            'sentiment_trend': [],
            'key_emotions': {},
            'risk_indicators': []
        }
        
        for text in texts:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            results['sentiment_scores'].append({
                'polarity': sentiment_score,
                'subjectivity': subjectivity,
                'length': len(text)
            })
        
        if results['sentiment_scores']:
            scores = [s['polarity'] for s in results['sentiment_scores']]
            results['overall_sentiment'] = np.mean(scores)
            results['sentiment_trend'] = scores  # Could add time-based analysis
        
        return results

    @staticmethod
    @tool("Analyze temporal patterns")
    def temporal_pattern_analysis(events: List[Dict]) -> Dict:
        """
        Analyze temporal patterns in events and mentions.
        
        Args:
            events: List of events with timestamps
        """
        analysis = {
            'event_frequency': {},
            'trend_analysis': {},
            'anomalies': [],
            'seasonal_patterns': {},
            'risk_timeline': []
        }
        
        # Group events by time periods
        event_dates = []
        for event in events:
            event_date = event.get('date')
            if isinstance(event_date, str):
                try:
                    event_date = datetime.fromisoformat(event_date)
                except:
                    continue
            if event_date:
                event_dates.append(event_date)
        
        if event_dates:
            # Calculate frequency by month
            month_counts = {}
            for date in event_dates:
                month_key = f"{date.year}-{date.month:02d}"
                month_counts[month_key] = month_counts.get(month_key, 0) + 1
            
            analysis['event_frequency'] = month_counts
            
            # Identify anomalies (months with unusually high activity)
            if len(month_counts) > 2:
                values = list(month_counts.values())
                mean_activity = np.mean(values)
                std_activity = np.std(values)
                
                for month, count in month_counts.items():
                    if count > mean_activity + 2 * std_activity:
                        analysis['anomalies'].append({
                            'month': month,
                            'activity': count,
                            'significance': 'high_activity_period'
                        })
        
        return analysis

    @staticmethod
    @tool("Compile comprehensive report")
    def compile_comprehensive_report(all_data: Dict) -> Dict:
        """
        Compile all analysis results into a comprehensive report.
        
        Args:
            all_data: Dictionary containing all analysis results
        """
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'analysis_version': '2.0',
                'target': all_data.get('target_info', {}),
                'data_sources': all_data.get('sources', [])
            },
            'executive_summary': {},
            'risk_assessment': {},
            'detailed_findings': {},
            'network_analysis': all_data.get('network_analysis', {}),
            'recommendations': [],
            'confidence_scores': all_data.get('confidence_scores', {}),
            'appendices': {}
        }
        
        # Generate executive summary
        report['executive_summary'] = AdvancedAnalysisTools._generate_executive_summary(all_data)
        
        # Compile risk assessment
        report['risk_assessment'] = {
            'financial_risk': all_data.get('financial_analysis', {}),
            'legal_risk': all_data.get('legal_analysis', {}),
            'reputational_risk': all_data.get('reputational_analysis', {}),
            'overall_risk_score': AdvancedAnalysisTools._calculate_overall_risk(all_data)
        }
        
        # Generate recommendations
        report['recommendations'] = AdvancedAnalysisTools._generate_comprehensive_recommendations(all_data)
        
        return report

    @staticmethod
    @tool("Generate executive summary") 
    def generate_executive_summary(data: Dict) -> str:
        """
        Generate an executive summary from analysis results.
        
        Args:
            data: Complete analysis data
        """
        return AdvancedAnalysisTools._generate_executive_summary(data)

    @staticmethod
    @tool("Create risk matrix")
    def create_risk_matrix(risks: Dict) -> Dict:
        """
        Create a comprehensive risk matrix visualization.
        
        Args:
            risks: Risk analysis results
        """
        matrix = {
            'high_probability_high_impact': [],
            'high_probability_low_impact': [],
            'low_probability_high_impact': [],
            'low_probability_low_impact': []
        }
        
        for risk_type, risk_data in risks.items():
            score = risk_data.get('score', 0)
            
            # Simple categorization - in reality this would be more sophisticated
            if score > 0.7:
                if 'financial' in risk_type or 'legal' in risk_type:
                    matrix['high_probability_high_impact'].append(risk_type)
                else:
                    matrix['high_probability_low_impact'].append(risk_type)
            else:
                if 'criminal' in risk_type:
                    matrix['low_probability_high_impact'].append(risk_type)
                else:
                    matrix['low_probability_low_impact'].append(risk_type)
        
        return matrix

    @staticmethod
    @tool("Generate actionable recommendations")
    def generate_recommendations(analysis_results: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis results.
        
        Args:
            analysis_results: Complete analysis results
        """
        return AdvancedAnalysisTools._generate_comprehensive_recommendations(analysis_results)

    @staticmethod
    @tool("Coordinate workflow between agents")
    def workflow_coordination(agent_states: Dict) -> Dict:
        """
        Coordinate workflow and task distribution between agents.
        
        Args:
            agent_states: Current states of all agents
        """
        coordination = {
            'next_actions': {},
            'priority_queue': [],
            'dependencies': {},
            'estimated_completion': None
        }
        
        # Simple workflow logic - expand as needed
        for agent_name, state in agent_states.items():
            if state.get('status') == 'waiting_for_data':
                coordination['priority_queue'].append({
                    'agent': agent_name,
                    'priority': 'high',
                    'action': 'provide_data'
                })
        
        return coordination

    @staticmethod
    @tool("Assess analysis quality")
    def quality_assessment(results: Dict) -> Dict:
        """
        Assess the quality and reliability of analysis results.
        
        Args:
            results: Analysis results to assess
        """
        quality_metrics = {
            'completeness': 0.0,
            'consistency': 0.0,
            'source_reliability': 0.0,
            'temporal_relevance': 0.0,
            'overall_quality': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Assess completeness
        required_sections = ['financial_risk', 'legal_risk', 'reputational_risk']
        completed_sections = sum(1 for section in required_sections 
                               if section in results and results[section])
        quality_metrics['completeness'] = completed_sections / len(required_sections)
        
        # Calculate overall quality
        metrics = [quality_metrics['completeness'], quality_metrics['consistency'],
                  quality_metrics['source_reliability'], quality_metrics['temporal_relevance']]
        quality_metrics['overall_quality'] = np.mean([m for m in metrics if m > 0])
        
        return quality_metrics

    @staticmethod
    @tool("Monitor analysis progress")
    def progress_monitoring(agents_status: Dict) -> Dict:
        """
        Monitor the progress of all agents in the analysis pipeline.
        
        Args:
            agents_status: Current status of all agents
        """
        progress = {
            'overall_progress': 0.0,
            'agent_progress': {},
            'bottlenecks': [],
            'estimated_completion': None,
            'performance_metrics': {}
        }
        
        total_progress = 0
        agent_count = len(agents_status)
        
        for agent_name, status in agents_status.items():
            agent_progress = status.get('progress', 0)
            progress['agent_progress'][agent_name] = agent_progress
            total_progress += agent_progress
            
            # Identify bottlenecks
            if agent_progress < 0.5 and status.get('status') == 'running':
                progress['bottlenecks'].append(agent_name)
        
        if agent_count > 0:
            progress['overall_progress'] = total_progress / agent_count
        
        return progress

    # Helper methods
    @staticmethod
    def _calculate_fuzzy_match(text1: str, text2: str) -> float:
        """Calculate fuzzy string matching score."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    @staticmethod
    def _analyze_context_relevance(text: str, target_info: Dict) -> float:
        """Analyze context relevance for entity disambiguation."""
        # Simple keyword-based relevance scoring
        keywords = target_info.get('keywords', [])
        if not keywords:
            return 0.5
        
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text.lower())
        return min(1.0, keyword_count / len(keywords))

    @staticmethod
    def _generate_executive_summary(data: Dict) -> str:
        """Generate executive summary from analysis data."""
        target_name = data.get('target_info', {}).get('name', 'Unknown Entity')
        
        summary_parts = [f"ADVERSE MEDIA ANALYSIS REPORT: {target_name}"]
        
        # Risk summary
        financial_risk = data.get('financial_analysis', {}).get('risk_score', 0)
        legal_risk = data.get('legal_analysis', {}).get('risk_score', 0) 
        reputational_risk = data.get('reputational_analysis', {}).get('risk_score', 0)
        
        risk_level = max(financial_risk, legal_risk, reputational_risk)
        if risk_level > 0.7:
            risk_summary = "HIGH RISK identified across multiple dimensions."
        elif risk_level > 0.4:
            risk_summary = "MODERATE RISK with specific areas of concern."
        else:
            risk_summary = "LOW RISK based on available information."
        
        summary_parts.append(risk_summary)
        
        return "\n".join(summary_parts)

    @staticmethod
    def _calculate_overall_risk(data: Dict) -> float:
        """Calculate overall risk score from all analysis components."""
        financial_risk = data.get('financial_analysis', {}).get('risk_score', 0)
        legal_risk = data.get('legal_analysis', {}).get('risk_score', 0)
        reputational_risk = data.get('reputational_analysis', {}).get('risk_score', 0)
        
        # Weighted average
        weights = [0.4, 0.35, 0.25]  # Financial, Legal, Reputational
        risks = [financial_risk, legal_risk, reputational_risk]
        
        return np.average(risks, weights=weights)

    @staticmethod
    def _generate_comprehensive_recommendations(data: Dict) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results."""
        recommendations = []
        
        # Financial recommendations
        financial_risk = data.get('financial_analysis', {}).get('risk_score', 0)
        if financial_risk > 0.6:
            recommendations.append(
                "IMMEDIATE: Conduct enhanced due diligence and financial audit. "
                "Consider engaging forensic accountants."
            )
        elif financial_risk > 0.3:
            recommendations.append(
                "Implement enhanced transaction monitoring and quarterly reviews."
            )
        
        # Legal recommendations
        legal_risk = data.get('legal_analysis', {}).get('risk_score', 0)
        if legal_risk > 0.5:
            recommendations.append(
                "URGENT: Seek legal counsel and review all compliance procedures. "
                "Consider regulatory reporting obligations."
            )
        
        # Reputational recommendations
        reputational_risk = data.get('reputational_analysis', {}).get('risk_score', 0)
        if reputational_risk > 0.4:
            recommendations.append(
                "Develop crisis communication plan and monitor public sentiment closely."
            )
        
        if not recommendations:
            recommendations.append(
                "Continue standard monitoring procedures. Schedule quarterly review."
            )
        
        return recommendations
