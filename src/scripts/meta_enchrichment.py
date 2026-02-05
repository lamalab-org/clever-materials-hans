from habanero import Crossref
from crossref_commons import retrieval
import pandas as pd
import numpy as np
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MetadataEnricher:
    """Enrich datasets with bibliographic metadata."""
    
    def __init__(self, email: Optional[str] = None, cache_file: Optional[str] = "cache/doi_metadata.json"):
        """Initialize with optional email for polite API usage."""
        self.email = email
        self.cache_file = Path(cache_file) if cache_file else None
 
        self.cr = Crossref(mailto=email) if email else Crossref()
 
        # Load cache
        self.cache = {}
        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        self.cache = json.load(f)
                except:
                    self.cache = {}
    
    def enrich_dataset(self, df: pd.DataFrame, doi_column: str = 'doi', 
                       batch_size: int = 50, max_dois: Optional[int] = None) -> pd.DataFrame:
        """Enrich dataframe with comprehensive metadata."""
        
        if doi_column not in df.columns:
            logger.warning(f"No DOI column '{doi_column}' found")
            return df
            
        logger.info(f"Enriching {df[doi_column].notna().sum()} DOIs with metadata...")
        
        # Initialize metadata columns
        metadata_columns = [
            'authors_full_list', 'author_first_name', 'author_last_name',
            'author_count', 'corresponding_author', 'author_orcids',
            'publication_year', 'publication_month', 'publication_day',
            'journal_name', 'journal_abbrev', 'journal_issn',
            'publisher', 'article_title', 'article_type',
            'citation_count', 'reference_count', 'altmetric_score',
            'institutions_list', 'institution_first', 'institution_countries',
            'funding_agencies', 'funding_count',
            'subject_areas', 'keywords',
            'open_access', 'license_type'
        ]
        
        for col in metadata_columns:
            if col not in df.columns:
                df[col] = None
        
        # Get unique DOIs
        unique_dois = df[doi_column].dropna().unique()
        if max_dois:
            unique_dois = unique_dois[:max_dois]
            
        # Process in batches
        for i in tqdm(range(0, len(unique_dois), batch_size), desc="Fetching metadata"):
            batch = unique_dois[i:i+batch_size]
            
            for doi in batch:
                if doi in self.cache:
                    metadata = self.cache[doi]
                else:
                    metadata = self._fetch_doi_metadata(doi)
                    self.cache[doi] = metadata
                    time.sleep(0.1)  # Rate limiting
                
                # Apply metadata to all rows with this DOI
                mask = df[doi_column] == doi
                for key, value in metadata.items():
                    if key in df.columns:
                        df.loc[mask, key] = value
        
        # Save cache
        if self.cache_file:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
                
        return df
    
    def _fetch_doi_metadata(self, doi: str) -> Dict[str, Any]:
        """Fetch comprehensive metadata for a single DOI."""
        
        metadata = {}
        
        try:
            # Clean the DOI
            doi = doi.strip()
            
            # Use the ids parameter for direct DOI lookup (not search)
            result = self.cr.works(ids=doi)
            
            if not result or 'message' not in result:
                logger.warning(f"No result for DOI: {doi}")
                return metadata
            
            msg = result['message']
            
            # Handle case where result is wrapped in items array
            if isinstance(msg, dict) and 'items' in msg and msg['items']:
                msg = msg['items'][0]
            
            # Authors - comprehensive extraction
            if 'author' in msg and msg['author']:
                authors = msg['author']
                
                # Full author list
                author_names = []
                orcids = []
                for author in authors:
                    name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                    if name:
                        author_names.append(name)
                    if 'ORCID' in author:
                        orcids.append(author['ORCID'].replace('http://orcid.org/', ''))
                
                metadata['authors_full_list'] = '; '.join(author_names)
                metadata['author_first_name'] = author_names[0] if author_names else None
                metadata['author_last_name'] = author_names[-1] if len(author_names) > 1 else None
                metadata['author_count'] = len(authors)
                metadata['author_orcids'] = '; '.join(orcids) if orcids else None
                
                # Corresponding author (often has email)
                for author in authors:
                    if author.get('email'):
                        metadata['corresponding_author'] = f"{author.get('given', '')} {author.get('family', '')}".strip()
                        break
                
                # Institutions
                institutions = []
                countries = []
                for author in authors:
                    if 'affiliation' in author:
                        for aff in author['affiliation']:
                            inst_name = aff.get('name', '')
                            if inst_name and inst_name not in institutions:
                                institutions.append(inst_name)
                                # Try to extract country
                                country = self._extract_country(inst_name)
                                if country and country not in countries:
                                    countries.append(country)
                
                metadata['institutions_list'] = '; '.join(institutions[:10])  # Limit to 10
                metadata['institution_first'] = institutions[0] if institutions else None
                metadata['institution_countries'] = '; '.join(countries)
            
            # Publication date - full extraction
            date_fields = ['published-print', 'published-online', 'created', 'deposited']
            for field in date_fields:
                if field in msg and msg[field] and 'date-parts' in msg[field]:
                    date_parts = msg[field]['date-parts'][0]
                    if date_parts:
                        metadata['publication_year'] = date_parts[0] if len(date_parts) > 0 else None
                        metadata['publication_month'] = date_parts[1] if len(date_parts) > 1 else None
                        metadata['publication_day'] = date_parts[2] if len(date_parts) > 2 else None
                        break
            
            # Journal information
            if 'container-title' in msg and msg['container-title']:
                metadata['journal_name'] = msg['container-title'][0]
                metadata['journal_abbrev'] = msg['container-title'][-1] if len(msg['container-title']) > 1 else None
            
            if 'ISSN' in msg and msg['ISSN']:
                metadata['journal_issn'] = msg['ISSN'][0]
            
            # Publisher
            metadata['publisher'] = msg.get('publisher', '')
            
            # Article information
            metadata['article_title'] = msg.get('title', [''])[0] if 'title' in msg else ''
            metadata['article_type'] = msg.get('type', '')
            
            # Citations and references
            metadata['citation_count'] = msg.get('is-referenced-by-count', 0)
            metadata['reference_count'] = msg.get('reference-count', 0)
            
            # Funding
            if 'funder' in msg and msg['funder']:
                funders = [f.get('name', '') for f in msg['funder'] if f.get('name')]
                metadata['funding_agencies'] = '; '.join(funders[:5])  # Limit to 5
                metadata['funding_count'] = len(msg['funder'])
            
            # Subject areas and keywords
            if 'subject' in msg:
                metadata['subject_areas'] = '; '.join(msg['subject'][:5])
            
            # Open access
            if 'license' in msg and msg['license']:
                metadata['open_access'] = True
                metadata['license_type'] = msg['license'][0].get('content-version', '') if msg['license'] else ''
            else:
                metadata['open_access'] = False
                
        except Exception as e:
            logger.debug(f"Error fetching metadata for {doi}: {e}")
            
        return metadata
    
    def _extract_country(self, institution: str) -> Optional[str]:
        """Extract country from institution name."""
        
        # Common country patterns
        countries = {
            'USA': ['USA', 'United States', 'America'],
            'UK': ['UK', 'United Kingdom', 'England', 'Scotland', 'Wales', 'Britain'],
            'China': ['China', 'Chinese'],
            'Germany': ['Germany', 'German'],
            'Japan': ['Japan', 'Japanese'],
            'France': ['France', 'French'],
            'Canada': ['Canada', 'Canadian'],
            'Australia': ['Australia', 'Australian'],
            'Singapore': ['Singapore'],
            'Switzerland': ['Switzerland', 'Swiss', 'ETH'],
            'Netherlands': ['Netherlands', 'Dutch'],
            'South Korea': ['Korea', 'Korean'],
            'India': ['India', 'Indian'],
            'Spain': ['Spain', 'Spanish'],
            'Italy': ['Italy', 'Italian']
        }
        
        inst_lower = institution.lower()
        for country, patterns in countries.items():
            for pattern in patterns:
                if pattern.lower() in inst_lower:
                    return country
                    
        return None