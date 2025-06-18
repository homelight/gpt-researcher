

## Copy paste the brightdata client from https://github.com/homelight/hl-docker-airflow/blob/main/dags/dw_dags/libs/web_researcher/brightdata_client.py

import requests
import os
import json
import urllib.parse as urlparse
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional


# Initialize logging
logger = logging.getLogger(__name__)

class BrightDataSearch:
    """
    Client for BrightData web search API
    """
    def __init__(self, query: str, zone: str = "hl_eng_serp_api1" ):
        """
        Initialize the BrightData client
        Args:
            api_token (str): BrightData API token
        """
        self.api_token = self.get_api_key()
        self.base_url = "https://api.brightdata.com/request"
        self.zone=zone
        self.query = query

    def get_api_key(self):
        """
        Gets the BrightData API key
        """
        try:
            api_key = os.environ["BRIGHTDATA_API_KEY"]
        except:
            raise Exception("Bright data API key not found. Please set the BRIGHTDATA_API_KEY environment variable. ")
        return api_key


    def search(self,max_results: int = 15, include_domains: Optional[List[str]] = None):
        """
        Search the web using BrightData
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            include_domains (list): List of domains to include in search results
            
        Returns:
            dict: Search results in a format similar to Tavily
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}"
        }
        
        # Build the complete query including domain filters if specified
        search_query = self.query
        if include_domains and len(include_domains) > 0:
            domain_filter = " " + " OR ".join([f"site:{domain}" for domain in include_domains])
            search_query += domain_filter
            
        # Encode the complete query for URL
        encoded_query = urlparse.quote(search_query)
        
        payload = {
            "zone": self.zone,
            "url": f"https://www.google.com/search?q={encoded_query}&brd_json=1&num={max_results}",
            "format": "json"
        }
        
        try:
            # logger.info(f"Sending BrightData search request for query: {self.query}")
            response = requests.post(self.base_url, headers=headers, json=payload)
            if 'error' in response.text.lower():
                if response.status_code != 200 and "No results for query" not in response.text:
                    raise Exception(f"BrightData search error: {response.text}")
                else:
                    logger.info(f"BrightData search issue, not an error: {response.text}")
            response.raise_for_status()
            
            # Parse the response
            body = response.json().get('body', '{}')
            if isinstance(body, str):
                data = json.loads(body)
            else:
                data = body
                
            # Convert to Tavily-like format
            results = []
            err_msg = ""
            if 'organic' in data:
                for item in data['organic'][:max_results]:
                    results.append({
                        'href': item.get('link', ''),
                        'title': item.get('title', ''),
                        'body': item.get('description', ''),
                        'score': 1.0 - (item.get('rank', 0) / max(max_results, 1)),  # Normalize score
                        'source': 'brightdata'
                    })
            
            if len(results) == 0:
                err_msg = f"BrightData search returned No results for query: {self.query}"
                logger.info(err_msg)
            
            logger.info(f"BrightData search returned {len(results)} results")


            return results, err_msg
            
        except Exception as e:
            logger.error(f"Error in BrightData search: {str(e)}")
            raise e