"""
Database File Parser Module
Handles parsing of various citation formats for systematic reviews
"""
import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatabaseFileParser:
    """
    Comprehensive parser for multiple citation database formats
    """
    
    def __init__(self):
        self.supported_formats = {
            'ris': self.parse_ris,
            'xml': self.parse_xml,
            'json': self.parse_json,
            'csv': self.parse_csv,
            'tsv': self.parse_tsv,
            'txt': self.parse_txt,
            'enw': self.parse_endnote,
            'bib': self.parse_bibtex
        }
    
    def parse_file(self, file_path: str, file_content: str, file_type: str = None) -> List[Dict[str, Any]]:
        """
        Main entry point for parsing files
        """
        if not file_type:
            file_type = file_path.split('.')[-1].lower()
        
        if file_type not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_type}")
        
        try:
            return self.supported_formats[file_type](file_content)
        except Exception as e:
            logger.error(f"Error parsing {file_type} file: {str(e)}")
            raise
    
    def parse_ris(self, content: str) -> List[Dict[str, Any]]:
        """Parse RIS format files"""
        citations = []
        current_citation = {}
        current_field = None
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            # Check if it's a new field
            if len(line) >= 6 and line[2:6] == '  - ':
                tag = line[:2]
                value = line[6:].strip()
                
                if tag == 'ER':  # End of record
                    if current_citation:
                        citations.append(self._normalize_citation(current_citation))
                        current_citation = {}
                        current_field = None
                else:
                    current_field = tag
                    if tag in ['AU', 'A1', 'A2', 'A3', 'A4']:  # Authors
                        if 'authors' not in current_citation:
                            current_citation['authors'] = []
                        current_citation['authors'].append(value)
                    elif tag in ['KW', 'DE']:  # Keywords
                        if 'keywords' not in current_citation:
                            current_citation['keywords'] = []
                        current_citation['keywords'].append(value)
                    else:
                        # Map RIS tags to standard fields
                        field_map = {
                            'TI': 'title',
                            'T1': 'title',
                            'TY': 'type',
                            'AB': 'abstract',
                            'JO': 'journal',
                            'JF': 'journal',
                            'PY': 'year',
                            'Y1': 'year',
                            'DO': 'doi',
                            'UR': 'url',
                            'VL': 'volume',
                            'IS': 'issue',
                            'SP': 'start_page',
                            'EP': 'end_page',
                            'SN': 'issn',
                            'ID': 'id'
                        }
                        if tag in field_map:
                            current_citation[field_map[tag]] = value
            elif current_field and line.startswith('  '):  # Continuation line
                # Append to the current field
                if current_field == 'AB' and 'abstract' in current_citation:
                    current_citation['abstract'] += ' ' + line.strip()
                elif current_field == 'TI' and 'title' in current_citation:
                    current_citation['title'] += ' ' + line.strip()
        
        # Don't forget the last citation if file doesn't end with ER
        if current_citation:
            citations.append(self._normalize_citation(current_citation))
        
        return citations
    
    def parse_xml(self, content: str) -> List[Dict[str, Any]]:
        """Parse XML format (PubMed, EndNote XML, etc.)"""
        citations = []
        
        try:
            root = ET.fromstring(content)
            
            # Try different XML structures
            # PubMed XML
            for article in root.findall('.//PubmedArticle'):
                citation = self._parse_pubmed_article(article)
                if citation:
                    citations.append(citation)
            
            # Generic XML records
            if not citations:
                for record in root.findall('.//record'):
                    citation = self._parse_generic_xml_record(record)
                    if citation:
                        citations.append(citation)
            
            # EndNote XML
            if not citations:
                for record in root.findall('.//xml-record'):
                    citation = self._parse_endnote_xml_record(record)
                    if citation:
                        citations.append(citation)
        
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            raise
        
        return citations
    
    def parse_json(self, content: str) -> List[Dict[str, Any]]:
        """Parse JSON format"""
        try:
            data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, list):
                return [self._normalize_citation(item) for item in data]
            elif isinstance(data, dict):
                if 'citations' in data:
                    return [self._normalize_citation(item) for item in data['citations']]
                elif 'results' in data:
                    return [self._normalize_citation(item) for item in data['results']]
                else:
                    # Single citation
                    return [self._normalize_citation(data)]
            else:
                raise ValueError("Unexpected JSON structure")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise
    
    def parse_csv(self, content: str) -> List[Dict[str, Any]]:
        """Parse CSV format"""
        try:
            # Use pandas for robust CSV parsing
            import io
            df = pd.read_csv(io.StringIO(content))
            
            citations = []
            for _, row in df.iterrows():
                citation = row.to_dict()
                # Convert NaN to None
                citation = {k: (None if pd.isna(v) else v) for k, v in citation.items()}
                citations.append(self._normalize_citation(citation))
            
            return citations
        
        except Exception as e:
            logger.error(f"CSV parsing error: {str(e)}")
            raise
    
    def parse_tsv(self, content: str) -> List[Dict[str, Any]]:
        """Parse TSV format"""
        try:
            import io
            df = pd.read_csv(io.StringIO(content), sep='\t')
            
            citations = []
            for _, row in df.iterrows():
                citation = row.to_dict()
                citation = {k: (None if pd.isna(v) else v) for k, v in citation.items()}
                citations.append(self._normalize_citation(citation))
            
            return citations
        
        except Exception as e:
            logger.error(f"TSV parsing error: {str(e)}")
            raise
    
    def parse_txt(self, content: str) -> List[Dict[str, Any]]:
        """Parse plain text format (PubMed format)"""
        citations = []
        current_citation = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line:
                if current_citation:
                    citations.append(self._normalize_citation(current_citation))
                    current_citation = {}
                continue
            
            # PubMed text format patterns
            if line.startswith('PMID:'):
                current_citation['pmid'] = line.replace('PMID:', '').strip()
            elif line.startswith('TI  -'):
                current_citation['title'] = line.replace('TI  -', '').strip()
            elif line.startswith('AB  -'):
                current_citation['abstract'] = line.replace('AB  -', '').strip()
            elif line.startswith('AU  -'):
                if 'authors' not in current_citation:
                    current_citation['authors'] = []
                current_citation['authors'].append(line.replace('AU  -', '').strip())
            elif line.startswith('TA  -'):
                current_citation['journal'] = line.replace('TA  -', '').strip()
            elif line.startswith('DP  -'):
                year_match = re.search(r'\d{4}', line)
                if year_match:
                    current_citation['year'] = int(year_match.group())
            elif line.startswith('DOI:'):
                current_citation['doi'] = line.replace('DOI:', '').strip()
        
        # Don't forget the last citation
        if current_citation:
            citations.append(self._normalize_citation(current_citation))
        
        return citations
    
    def parse_endnote(self, content: str) -> List[Dict[str, Any]]:
        """Parse EndNote tagged format"""
        citations = []
        current_citation = {}
        
        for line in content.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('%0'):  # Reference type
                if current_citation:
                    citations.append(self._normalize_citation(current_citation))
                current_citation = {'type': line[3:].strip()}
            elif line.startswith('%'):
                tag = line[1]
                value = line[3:].strip() if len(line) > 3 else ''
                
                field_map = {
                    'T': 'title',
                    'A': 'authors',
                    'D': 'year',
                    'J': 'journal',
                    'V': 'volume',
                    'N': 'issue',
                    'P': 'pages',
                    'X': 'abstract',
                    'K': 'keywords',
                    'U': 'url',
                    'R': 'doi'
                }
                
                if tag in field_map:
                    field = field_map[tag]
                    if field in ['authors', 'keywords']:
                        if field not in current_citation:
                            current_citation[field] = []
                        current_citation[field].append(value)
                    else:
                        current_citation[field] = value
        
        # Don't forget the last citation
        if current_citation:
            citations.append(self._normalize_citation(current_citation))
        
        return citations
    
    def parse_bibtex(self, content: str) -> List[Dict[str, Any]]:
        """Parse BibTeX format"""
        citations = []
        
        # Regex to find BibTeX entries
        entry_pattern = r'@(\w+)\s*{\s*([^,]+),\s*((?:[^{}]|{[^}]*})*)\}'
        entries = re.findall(entry_pattern, content, re.DOTALL)
        
        for entry_type, cite_key, fields in entries:
            citation = {
                'type': entry_type,
                'cite_key': cite_key
            }
            
            # Parse fields
            field_pattern = r'(\w+)\s*=\s*{([^}]*)}'
            for field_name, field_value in re.findall(field_pattern, fields):
                field_name = field_name.lower()
                
                if field_name == 'author':
                    # Split authors by 'and'
                    citation['authors'] = [a.strip() for a in field_value.split(' and ')]
                elif field_name == 'keywords':
                    citation['keywords'] = [k.strip() for k in field_value.split(',')]
                else:
                    citation[field_name] = field_value.strip()
            
            citations.append(self._normalize_citation(citation))
        
        return citations
    
    def _normalize_citation(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize citation data to a standard format"""
        normalized = {
            'citation_id': citation.get('id', self._generate_citation_id(citation)),
            'title': citation.get('title', ''),
            'abstract': citation.get('abstract', ''),
            'authors': self._normalize_authors(citation.get('authors', [])),
            'journal': citation.get('journal', ''),
            'year': self._normalize_year(citation.get('year')),
            'doi': citation.get('doi', ''),
            'pmid': citation.get('pmid', ''),
            'keywords': self._normalize_keywords(citation.get('keywords', [])),
            'url': citation.get('url', ''),
            'volume': citation.get('volume', ''),
            'issue': citation.get('issue', ''),
            'pages': self._normalize_pages(citation),
            'type': citation.get('type', 'article'),
            'source_database': citation.get('source_database', 'unknown'),
            'raw_data': json.dumps(citation)
        }
        
        return normalized
    
    def _normalize_authors(self, authors: Any) -> str:
        """Normalize author field"""
        if isinstance(authors, list):
            return '; '.join([str(a).strip() for a in authors])
        elif isinstance(authors, str):
            return authors
        else:
            return ''
    
    def _normalize_year(self, year: Any) -> Optional[int]:
        """Normalize year field"""
        if isinstance(year, int):
            return year
        elif isinstance(year, str):
            # Extract first 4-digit number
            match = re.search(r'\d{4}', year)
            if match:
                return int(match.group())
        return None
    
    def _normalize_keywords(self, keywords: Any) -> str:
        """Normalize keywords field"""
        if isinstance(keywords, list):
            return '; '.join([str(k).strip() for k in keywords])
        elif isinstance(keywords, str):
            return keywords
        else:
            return ''
    
    def _normalize_pages(self, citation: Dict[str, Any]) -> str:
        """Normalize page numbers"""
        if 'pages' in citation:
            return str(citation['pages'])
        elif 'start_page' in citation and 'end_page' in citation:
            return f"{citation['start_page']}-{citation['end_page']}"
        elif 'start_page' in citation:
            return str(citation['start_page'])
        else:
            return ''
    
    def _generate_citation_id(self, citation: Dict[str, Any]) -> str:
        """Generate a unique ID for a citation"""
        # Use DOI if available
        if citation.get('doi'):
            return f"doi_{citation['doi'].replace('/', '_')}"
        # Use PMID if available
        elif citation.get('pmid'):
            return f"pmid_{citation['pmid']}"
        # Generate from title and year
        else:
            title = citation.get('title', 'unknown')
            year = citation.get('year', 'unknown')
            # Create a simple hash
            title_words = title.lower().split()[:3]
            title_part = '_'.join(title_words)
            return f"{title_part}_{year}_{hash(title) % 10000}"
    
    def _parse_pubmed_article(self, article: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a PubMed XML article element"""
        try:
            citation = {}
            
            # Extract PMID
            pmid = article.find('.//PMID')
            if pmid is not None:
                citation['pmid'] = pmid.text
            
            # Extract article details
            article_elem = article.find('.//Article')
            if article_elem is not None:
                # Title
                title = article_elem.find('.//ArticleTitle')
                if title is not None:
                    citation['title'] = title.text
                
                # Abstract
                abstract = article_elem.find('.//AbstractText')
                if abstract is not None:
                    citation['abstract'] = abstract.text
                
                # Journal
                journal = article_elem.find('.//Journal/Title')
                if journal is not None:
                    citation['journal'] = journal.text
                
                # Year
                year = article_elem.find('.//PubDate/Year')
                if year is not None:
                    citation['year'] = year.text
                
                # Authors
                authors = []
                for author in article_elem.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None:
                        name = last_name.text
                        if first_name is not None:
                            name = f"{name}, {first_name.text}"
                        authors.append(name)
                citation['authors'] = authors
            
            return citation
        
        except Exception as e:
            logger.error(f"Error parsing PubMed article: {str(e)}")
            return None
    
    def _parse_generic_xml_record(self, record: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a generic XML record"""
        citation = {}
        
        # Try to extract common fields
        for field in ['title', 'abstract', 'authors', 'journal', 'year', 'doi', 'pmid']:
            elem = record.find(f'.//{field}')
            if elem is not None and elem.text:
                citation[field] = elem.text
        
        return citation if citation else None
    
    def _parse_endnote_xml_record(self, record: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse an EndNote XML record"""
        citation = {}
        
        # Map of EndNote XML fields to standard fields
        field_map = {
            'title': 'title',
            'abstract': 'abstract',
            'author': 'authors',
            'secondary-title': 'journal',
            'year': 'year',
            'doi': 'doi',
            'accession-number': 'pmid'
        }
        
        for endnote_field, standard_field in field_map.items():
            elem = record.find(f'.//{endnote_field}')
            if elem is not None and elem.text:
                citation[standard_field] = elem.text
        
        return citation if citation else None