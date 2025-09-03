"""
External API Connectors for AMP

Real integrations with external data sources including news APIs, 
social media platforms, and public databases.
"""

import asyncio
import aiohttp
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time
from urllib.parse import quote_plus, urljoin
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    success: bool
    data: List[Dict]
    error: Optional[str] = None
    metadata: Optional[Dict] = None
    rate_limit_remaining: Optional[int] = None

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def can_make_call(self) -> bool:
        """Check if we can make a call without exceeding rate limit."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        return len(self.calls) < self.calls_per_minute
    
    def record_call(self):
        """Record that we made a call."""
        self.calls.append(time.time())
    
    async def wait_if_needed(self):
        """Wait if we're at the rate limit."""
        if not self.can_make_call():
            wait_time = 60 - (time.time() - min(self.calls))
            logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)

class NewsAPIConnector:
    """Connector for NewsAPI.org and similar news services."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter(1000)  # 1000 requests per day for free tier
        
    async def search_everything(self, query: str, language: str = 'en', 
                              sort_by: str = 'publishedAt', page_size: int = 100,
                              from_date: datetime = None, to_date: datetime = None) -> APIResponse:
        """
        Search news articles using NewsAPI Everything endpoint.
        
        Args:
            query: Search query
            language: Language code (en, pt, etc.)
            sort_by: Sort order (publishedAt, relevancy, popularity)
            page_size: Number of articles per page (max 100)
            from_date: Search from this date
            to_date: Search until this date
            
        Returns:
            APIResponse with articles
        """
        await self.rate_limiter.wait_if_needed()
        
        params = {
            'q': query,
            'language': language,
            'sortBy': sort_by,
            'pageSize': page_size,
            'apiKey': self.api_key
        }
        
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%dT%H:%M:%S')
        if to_date:
            params['to'] = to_date.strftime('%Y-%m-%dT%H:%M:%S')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/everything", params=params) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        data = await response.json()
                        articles = []
                        
                        for article in data.get('articles', []):
                            articles.append({
                                'title': article.get('title'),
                                'content': article.get('content') or article.get('description', ''),
                                'source': article.get('source', {}).get('name'),
                                'url': article.get('url'),
                                'published_date': article.get('publishedAt'),
                                'author': article.get('author'),
                                'relevance_score': 1.0  # NewsAPI doesn't provide relevance scores
                            })
                        
                        return APIResponse(
                            success=True,
                            data=articles,
                            metadata={
                                'total_results': data.get('totalResults', 0),
                                'page_size': page_size
                            }
                        )
                    else:
                        error_data = await response.json()
                        return APIResponse(
                            success=False,
                            data=[],
                            error=error_data.get('message', f'HTTP {response.status}')
                        )
                        
        except Exception as e:
            logger.error(f"NewsAPI error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )

class GoogleNewsConnector:
    """Connector for Google News RSS feeds."""
    
    def __init__(self):
        self.base_url = "https://news.google.com/rss"
        self.rate_limiter = RateLimiter(100)  # Conservative rate limit
    
    async def search_news(self, query: str, language: str = 'en', 
                         country: str = 'US', max_results: int = 50) -> APIResponse:
        """
        Search Google News via RSS feeds.
        
        Args:
            query: Search query
            language: Language code
            country: Country code
            max_results: Maximum number of results
            
        Returns:
            APIResponse with news articles
        """
        await self.rate_limiter.wait_if_needed()
        
        # Construct Google News RSS URL
        encoded_query = quote_plus(query)
        url = f"{self.base_url}/search?q={encoded_query}&hl={language}&gl={country}&ceid={country}:{language}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        xml_content = await response.text()
                        root = ET.fromstring(xml_content)
                        
                        articles = []
                        items = root.findall('.//item')[:max_results]
                        
                        for item in items:
                            title = item.find('title')
                            link = item.find('link') 
                            description = item.find('description')
                            pub_date = item.find('pubDate')
                            source = item.find('source')
                            
                            articles.append({
                                'title': title.text if title is not None else '',
                                'content': description.text if description is not None else '',
                                'source': source.text if source is not None else 'Google News',
                                'url': link.text if link is not None else '',
                                'published_date': pub_date.text if pub_date is not None else '',
                                'relevance_score': 0.8  # Default relevance for RSS results
                            })
                        
                        return APIResponse(
                            success=True,
                            data=articles,
                            metadata={'source': 'google_news_rss'}
                        )
                    else:
                        return APIResponse(
                            success=False,
                            data=[],
                            error=f'HTTP {response.status}'
                        )
                        
        except Exception as e:
            logger.error(f"Google News RSS error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )

class TwitterAPIConnector:
    """Connector for Twitter API v2 (requires Twitter API access)."""
    
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.rate_limiter = RateLimiter(300)  # 300 requests per 15-minute window
        
    async def search_tweets(self, query: str, max_results: int = 100,
                           start_time: datetime = None, end_time: datetime = None) -> APIResponse:
        """
        Search recent tweets using Twitter API v2.
        
        Args:
            query: Search query (Twitter search syntax)
            max_results: Maximum number of tweets (10-100)
            start_time: Search from this time
            end_time: Search until this time
            
        Returns:
            APIResponse with tweets
        """
        await self.rate_limiter.wait_if_needed()
        
        headers = {
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'query': query,
            'max_results': min(max_results, 100),
            'tweet.fields': 'created_at,author_id,public_metrics,context_annotations',
            'expansions': 'author_id',
            'user.fields': 'username,name,verified'
        }
        
        if start_time:
            params['start_time'] = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        if end_time:
            params['end_time'] = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/tweets/search/recent", 
                                     headers=headers, params=params) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        data = await response.json()
                        tweets = []
                        
                        # Build user lookup
                        users = {}
                        if 'includes' in data and 'users' in data['includes']:
                            users = {user['id']: user for user in data['includes']['users']}
                        
                        for tweet in data.get('data', []):
                            author_id = tweet.get('author_id')
                            user = users.get(author_id, {})
                            
                            tweets.append({
                                'platform': 'twitter',
                                'content': tweet.get('text', ''),
                                'author': f"@{user.get('username', 'unknown')}",
                                'author_name': user.get('name', ''),
                                'author_verified': user.get('verified', False),
                                'engagement': tweet.get('public_metrics', {}),
                                'timestamp': tweet.get('created_at'),
                                'url': f"https://twitter.com/i/status/{tweet.get('id')}",
                                'sentiment': 'neutral'  # Would need sentiment analysis
                            })
                        
                        return APIResponse(
                            success=True,
                            data=tweets,
                            metadata={
                                'result_count': data.get('meta', {}).get('result_count', 0)
                            },
                            rate_limit_remaining=response.headers.get('x-rate-limit-remaining')
                        )
                    else:
                        error_data = await response.json()
                        return APIResponse(
                            success=False,
                            data=[],
                            error=error_data.get('title', f'HTTP {response.status}')
                        )
                        
        except Exception as e:
            logger.error(f"Twitter API error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )

class RedditConnector:
    """Connector for Reddit API (read-only, no authentication required)."""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.rate_limiter = RateLimiter(60)  # 60 requests per minute
        
    async def search_subreddits(self, query: str, subreddits: List[str] = None,
                               sort: str = 'relevance', time: str = 'all',
                               limit: int = 100) -> APIResponse:
        """
        Search Reddit posts across specified subreddits.
        
        Args:
            query: Search query
            subreddits: List of subreddit names (without r/)
            sort: Sort order (relevance, hot, top, new)
            time: Time filter (all, year, month, week, day)
            limit: Maximum results
            
        Returns:
            APIResponse with Reddit posts
        """
        await self.rate_limiter.wait_if_needed()
        
        headers = {
            'User-Agent': 'AMP-Bot/1.0 (Adverse Media Profiler)'
        }
        
        posts = []
        
        # If no subreddits specified, search all of Reddit
        if not subreddits:
            subreddits = ['all']
        
        for subreddit in subreddits[:5]:  # Limit to 5 subreddits to avoid rate limits
            url = f"{self.base_url}/r/{subreddit}/search.json"
            params = {
                'q': query,
                'sort': sort,
                't': time,
                'limit': min(limit // len(subreddits), 25),  # Distribute limit across subreddits
                'restrict_sr': 1
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        self.rate_limiter.record_call()
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            for post_data in data.get('data', {}).get('children', []):
                                post = post_data.get('data', {})
                                
                                posts.append({
                                    'platform': 'reddit',
                                    'content': f"{post.get('title', '')} {post.get('selftext', '')}",
                                    'title': post.get('title', ''),
                                    'author': f"u/{post.get('author', 'unknown')}",
                                    'subreddit': f"r/{post.get('subreddit', '')}",
                                    'score': post.get('score', 0),
                                    'num_comments': post.get('num_comments', 0),
                                    'timestamp': datetime.fromtimestamp(post.get('created_utc', 0)),
                                    'url': f"https://reddit.com{post.get('permalink', '')}",
                                    'engagement': {
                                        'upvotes': post.get('ups', 0),
                                        'downvotes': post.get('downs', 0),
                                        'comments': post.get('num_comments', 0)
                                    }
                                })
                        else:
                            logger.warning(f"Reddit API error for r/{subreddit}: HTTP {response.status}")
                            
            except Exception as e:
                logger.error(f"Reddit search error for r/{subreddit}: {str(e)}")
                continue
        
        return APIResponse(
            success=True,
            data=posts,
            metadata={'subreddits_searched': subreddits, 'total_posts': len(posts)}
        )

class OFACConnector:
    """Connector for OFAC (Office of Foreign Assets Control) sanctions data."""
    
    def __init__(self):
        self.base_url = "https://www.treasury.gov/ofac/downloads"
        self.rate_limiter = RateLimiter(30)  # Conservative rate limit
        
    async def get_sdn_list(self) -> APIResponse:
        """
        Download and parse OFAC Specially Designated Nationals (SDN) list.
        
        Returns:
            APIResponse with sanctions data
        """
        await self.rate_limiter.wait_if_needed()
        
        sdn_url = f"{self.base_url}/sdn.csv"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sdn_url) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        csv_content = await response.text()
                        lines = csv_content.strip().split('\n')
                        
                        sanctions_data = []
                        for line in lines[1:]:  # Skip header
                            fields = line.split(',')
                            if len(fields) >= 6:
                                sanctions_data.append({
                                    'name': fields[1].strip('"'),
                                    'type': fields[2].strip('"'),
                                    'program': fields[3].strip('"'),
                                    'title': fields[4].strip('"'),
                                    'call_sign': fields[5].strip('"'),
                                    'vessel_type': fields[6].strip('"') if len(fields) > 6 else '',
                                    'tonnage': fields[7].strip('"') if len(fields) > 7 else '',
                                    'gross_registered_tonnage': fields[8].strip('"') if len(fields) > 8 else '',
                                    'vessel_flag': fields[9].strip('"') if len(fields) > 9 else '',
                                    'vessel_owner': fields[10].strip('"') if len(fields) > 10 else '',
                                    'remarks': fields[11].strip('"') if len(fields) > 11 else ''
                                })
                        
                        return APIResponse(
                            success=True,
                            data=sanctions_data,
                            metadata={'total_entries': len(sanctions_data), 'source': 'OFAC_SDN'}
                        )
                    else:
                        return APIResponse(
                            success=False,
                            data=[],
                            error=f'HTTP {response.status}'
                        )
                        
        except Exception as e:
            logger.error(f"OFAC API error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )
    
    def search_sdn_local(self, query: str, sdn_data: List[Dict]) -> List[Dict]:
        """
        Search locally cached SDN data.
        
        Args:
            query: Search query (name)
            sdn_data: SDN data from get_sdn_list()
            
        Returns:
            List of matching sanctions records
        """
        matches = []
        query_lower = query.lower()
        
        for record in sdn_data:
            name = record.get('name', '').lower()
            if query_lower in name or self._fuzzy_match(query_lower, name):
                matches.append(record)
        
        return matches
    
    def _fuzzy_match(self, query: str, name: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for names."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, query, name).ratio() > threshold

class SerperAPIConnector:
    """Connector for Serper.dev Google Search API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"
        self.rate_limiter = RateLimiter(100)  # 100 requests per day for free tier
        
    async def search_news(self, query: str, num_results: int = 50,
                         country: str = 'us', language: str = 'en') -> APIResponse:
        """
        Search news using Serper Google News API.
        
        Args:
            query: Search query
            num_results: Number of results (max 100)
            country: Country code
            language: Language code
            
        Returns:
            APIResponse with news articles
        """
        await self.rate_limiter.wait_if_needed()
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        data = {
            'q': query,
            'num': min(num_results, 100),
            'gl': country,
            'hl': language
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/news", 
                                       headers=headers, json=data) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        result = await response.json()
                        articles = []
                        
                        for article in result.get('news', []):
                            articles.append({
                                'title': article.get('title'),
                                'content': article.get('snippet', ''),
                                'source': article.get('source'),
                                'url': article.get('link'),
                                'published_date': article.get('date'),
                                'relevance_score': 0.9,  # High relevance from Google
                                'image_url': article.get('imageUrl')
                            })
                        
                        return APIResponse(
                            success=True,
                            data=articles,
                            metadata={
                                'search_metadata': result.get('searchParameters', {}),
                                'total_results': len(articles)
                            }
                        )
                    else:
                        return APIResponse(
                            success=False,
                            data=[],
                            error=f'HTTP {response.status}'
                        )
                        
        except Exception as e:
            logger.error(f"Serper API error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )

class CorporateRegistryConnector:
    """Connector for various corporate registries and business databases."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter(60)
        
    async def search_sec_filings(self, company_name: str, cik: str = None) -> APIResponse:
        """
        Search SEC EDGAR database for company filings.
        
        Args:
            company_name: Company name
            cik: Central Index Key (if known)
            
        Returns:
            APIResponse with SEC filings
        """
        await self.rate_limiter.wait_if_needed()
        
        base_url = "https://data.sec.gov/submissions"
        headers = {
            'User-Agent': 'AMP-Bot contact@example.com',  # SEC requires User-Agent
            'Accept': 'application/json'
        }
        
        try:
            # If we have CIK, search directly
            if cik:
                url = f"{base_url}/CIK{cik.zfill(10)}.json"
            else:
                # Search for company first
                search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
                # This would require parsing HTML, simplified for now
                return APIResponse(
                    success=False,
                    data=[],
                    error="Company name search not implemented - CIK required"
                )
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    self.rate_limiter.record_call()
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        filings = []
                        recent_filings = data.get('filings', {}).get('recent', {})
                        
                        for i in range(len(recent_filings.get('form', [])[:50])):  # Last 50 filings
                            filing = {
                                'form_type': recent_filings['form'][i],
                                'filing_date': recent_filings['filingDate'][i],
                                'report_date': recent_filings['reportDate'][i],
                                'accession_number': recent_filings['accessionNumber'][i],
                                'primary_document': recent_filings['primaryDocument'][i],
                                'url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{recent_filings['accessionNumber'][i].replace('-', '')}/{recent_filings['primaryDocument'][i]}"
                            }
                            filings.append(filing)
                        
                        return APIResponse(
                            success=True,
                            data=filings,
                            metadata={
                                'company_name': data.get('name'),
                                'cik': data.get('cik'),
                                'sic': data.get('sic'),
                                'industry': data.get('sicDescription')
                            }
                        )
                    else:
                        return APIResponse(
                            success=False,
                            data=[],
                            error=f'HTTP {response.status}'
                        )
                        
        except Exception as e:
            logger.error(f"SEC API error: {str(e)}")
            return APIResponse(
                success=False,
                data=[],
                error=str(e)
            )

class ExternalAPIManager:
    """Manager class for coordinating all external API connectors."""
    
    def __init__(self, config: Dict[str, str]):
        """
        Initialize API manager with configuration.
        
        Args:
            config: Dictionary containing API keys and settings
        """
        self.config = config
        self.connectors = {}
        
        # Initialize available connectors
        if config.get('news_api_key'):
            self.connectors['newsapi'] = NewsAPIConnector(config['news_api_key'])
        
        self.connectors['google_news'] = GoogleNewsConnector()
        
        if config.get('twitter_bearer_token'):
            self.connectors['twitter'] = TwitterAPIConnector(config['twitter_bearer_token'])
        
        self.connectors['reddit'] = RedditConnector()
        self.connectors['ofac'] = OFACConnector()
        
        if config.get('serper_api_key'):
            self.connectors['serper'] = SerperAPIConnector(config['serper_api_key'])
        
        self.connectors['corporate'] = CorporateRegistryConnector()
        
        logger.info(f"Initialized {len(self.connectors)} API connectors: {list(self.connectors.keys())}")
    
    async def comprehensive_search(self, target: str, search_config: Dict = None) -> Dict[str, APIResponse]:
        """
        Perform comprehensive search across all available APIs.
        
        Args:
            target: Target entity to search for
            search_config: Search configuration options
            
        Returns:
            Dictionary mapping connector names to APIResponse objects
        """
        if search_config is None:
            search_config = {}
        
        # Define search tasks
        search_tasks = {}
        
        # News searches
        if 'newsapi' in self.connectors:
            search_tasks['newsapi'] = self.connectors['newsapi'].search_everything(
                query=target,
                page_size=search_config.get('news_limit', 50)
            )
        
        if 'google_news' in self.connectors:
            search_tasks['google_news'] = self.connectors['google_news'].search_news(
                query=target,
                max_results=search_config.get('news_limit', 50)
            )
        
        if 'serper' in self.connectors:
            search_tasks['serper_news'] = self.connectors['serper'].search_news(
                query=target,
                num_results=search_config.get('news_limit', 50)
            )
        
        # Social media searches
        if 'twitter' in self.connectors:
            search_tasks['twitter'] = self.connectors['twitter'].search_tweets(
                query=target,
                max_results=search_config.get('social_limit', 100)
            )
        
        if 'reddit' in self.connectors:
            search_tasks['reddit'] = self.connectors['reddit'].search_subreddits(
                query=target,
                limit=search_config.get('social_limit', 50)
            )
        
        # Execute all searches concurrently
        logger.info(f"Starting comprehensive search for '{target}' across {len(search_tasks)} APIs")
        
        results = {}
        completed_tasks = await asyncio.gather(*search_tasks.values(), return_exceptions=True)
        
        for i, (connector_name, task_result) in enumerate(zip(search_tasks.keys(), completed_tasks)):
            if isinstance(task_result, Exception):
                logger.error(f"Search failed for {connector_name}: {str(task_result)}")
                results[connector_name] = APIResponse(
                    success=False,
                    data=[],
                    error=str(task_result)
                )
            else:
                results[connector_name] = task_result
                logger.info(f"Search completed for {connector_name}: {len(task_result.data)} results")
        
        return results
    
    async def check_sanctions_lists(self, target: str) -> Dict[str, List[Dict]]:
        """
        Check target against various sanctions lists.
        
        Args:
            target: Target entity to check
            
        Returns:
            Dictionary of sanctions matches by source
        """
        sanctions_results = {}
        
        # Check OFAC SDN list
        if 'ofac' in self.connectors:
            try:
                ofac_response = await self.connectors['ofac'].get_sdn_list()
                if ofac_response.success:
                    matches = self.connectors['ofac'].search_sdn_local(target, ofac_response.data)
                    sanctions_results['ofac_sdn'] = matches
                    logger.info(f"OFAC check completed: {len(matches)} matches found")
                else:
                    logger.error(f"OFAC SDN list download failed: {ofac_response.error}")
                    sanctions_results['ofac_sdn'] = []
            except Exception as e:
                logger.error(f"OFAC sanctions check failed: {str(e)}")
                sanctions_results['ofac_sdn'] = []
        
        return sanctions_results
    
    def get_available_connectors(self) -> List[str]:
        """Get list of available connector names."""
        return list(self.connectors.keys())
    
    def get_connector_status(self) -> Dict[str, Dict]:
        """Get status information for all connectors."""
        status = {}
        
        for name, connector in self.connectors.items():
            status[name] = {
                'available': True,
                'rate_limit_remaining': getattr(connector, 'rate_limiter', None) and 
                                       connector.rate_limiter.can_make_call(),
                'connector_type': type(connector).__name__
            }
        
        return status

# Factory function for easy initialization
def create_api_manager_from_env() -> ExternalAPIManager:
    """
    Create API manager from environment variables.
    
    Returns:
        ExternalAPIManager instance configured with environment variables
    """
    config = {
        'news_api_key': os.getenv('NEWS_API_KEY'),
        'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'serper_api_key': os.getenv('SERPER_API_KEY'),
    }
    
    return ExternalAPIManager(config)
