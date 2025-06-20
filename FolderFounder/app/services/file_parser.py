"""
File parsing service for various citation formats
"""
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from datetime import datetime


class CitationParser:
    """Parser for various citation file formats"""
    
    @staticmethod
    def parse_file(filename: str, content: str) -> List[Dict[str, Any]]:
        """Parse citation file based on extension"""
        ext = filename.lower().split('.')[-1]
        
        if ext == 'ris':
            return CitationParser.parse_ris(content)
        elif ext == 'xml':
            return CitationParser.parse_xml(content)
        elif ext == 'enw':
            return CitationParser.parse_endnote(content)
        elif ext == 'bib':
            return CitationParser.parse_bibtex(content)
        elif ext in ['rdf', 'json']:
            return CitationParser.parse_zotero(content, ext)
        elif ext == 'csv':
            return CitationParser.parse_csv(content)
        elif ext == 'txt':
            # Assume PubMed format for .txt files
            return CitationParser.parse_pubmed_txt(content)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def parse_ris(content: str) -> List[Dict[str, Any]]:
        """Parse RIS format files"""
        citations = []
        entries = content.split('ER  -')
        
        for entry in entries:
            if not entry.strip():
                continue
                
            citation = {
                "title": "",
                "authors": "",
                "journal": "",
                "year": None,
                "abstract": "",
                "doi": "",
                "keywords": "",
                "relevance_score": 0.5
            }
            
            lines = entry.strip().split('\n')
            authors_list = []
            keywords_list = []
            abstract_lines = []
            current_field = None
            
            for line in lines:
                if line.startswith('TY  - '):
                    citation["type"] = line[6:].strip()
                elif line.startswith('TI  - ') or line.startswith('T1  - '):
                    citation["title"] = line[6:].strip()
                    current_field = "title"
                elif line.startswith('AU  - ') or line.startswith('A1  - '):
                    authors_list.append(line[6:].strip())
                    current_field = "author"
                elif line.startswith('JO  - ') or line.startswith('JF  - '):
                    citation["journal"] = line[6:].strip()
                    current_field = "journal"
                elif line.startswith('PY  - ') or line.startswith('Y1  - '):
                    year_str = line[6:].strip()
                    if year_str and year_str[:4].isdigit():
                        citation["year"] = int(year_str[:4])
                    current_field = None
                elif line.startswith('AB  - '):
                    abstract_lines.append(line[6:].strip())
                    current_field = "abstract"
                elif line.startswith('KW  - '):
                    keywords_list.append(line[6:].strip())
                    current_field = "keyword"
                elif line.startswith('DO  - '):
                    citation["doi"] = line[6:].strip()
                    current_field = None
                elif line.startswith('  ') and current_field:
                    # Continuation of previous field
                    if current_field == "title":
                        citation["title"] += " " + line.strip()
                    elif current_field == "abstract":
                        abstract_lines.append(line.strip())
                    elif current_field == "journal":
                        citation["journal"] += " " + line.strip()
            
            citation["authors"] = "; ".join(authors_list)
            citation["abstract"] = " ".join(abstract_lines)
            citation["keywords"] = "; ".join(keywords_list)
            
            # Calculate relevance score based on completeness
            citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
            
            if citation["title"]:  # Only add if there's at least a title
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def parse_xml(content: str) -> List[Dict[str, Any]]:
        """Parse XML format (PubMed, EndNote XML, etc.)"""
        citations = []
        
        try:
            root = ET.fromstring(content)
            
            # Try different XML structures
            # PubMed XML
            articles = root.findall('.//PubmedArticle') or root.findall('.//Article')
            
            for article in articles:
                citation = CitationParser._extract_pubmed_xml(article)
                if citation["title"]:
                    citations.append(citation)
            
            # If no PubMed articles found, try generic XML structure
            if not citations:
                records = root.findall('.//record') or root.findall('.//citation')
                for record in records:
                    citation = CitationParser._extract_generic_xml(record)
                    if citation["title"]:
                        citations.append(citation)
        
        except ET.ParseError:
            # If XML parsing fails, return empty list
            pass
        
        return citations
    
    @staticmethod
    def parse_endnote(content: str) -> List[Dict[str, Any]]:
        """Parse EndNote tagged format"""
        citations = []
        entries = re.split(r'\n(?=%0)', content)
        
        for entry in entries:
            if not entry.strip():
                continue
                
            citation = {
                "title": "",
                "authors": "",
                "journal": "",
                "year": None,
                "abstract": "",
                "doi": "",
                "keywords": "",
                "relevance_score": 0.5
            }
            
            lines = entry.strip().split('\n')
            authors_list = []
            keywords_list = []
            
            for line in lines:
                if line.startswith('%T '):
                    citation["title"] = line[3:].strip()
                elif line.startswith('%A '):
                    authors_list.append(line[3:].strip())
                elif line.startswith('%J '):
                    citation["journal"] = line[3:].strip()
                elif line.startswith('%D '):
                    year_str = line[3:].strip()
                    if year_str and year_str[:4].isdigit():
                        citation["year"] = int(year_str[:4])
                elif line.startswith('%X '):
                    citation["abstract"] = line[3:].strip()
                elif line.startswith('%K '):
                    keywords_list.append(line[3:].strip())
                elif line.startswith('%R '):
                    citation["doi"] = line[3:].strip()
            
            citation["authors"] = "; ".join(authors_list)
            citation["keywords"] = "; ".join(keywords_list)
            citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
            
            if citation["title"]:
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def parse_bibtex(content: str) -> List[Dict[str, Any]]:
        """Parse BibTeX format"""
        citations = []
        entries = re.findall(r'@\w+\{[^@]+\}', content, re.DOTALL)
        
        for entry in entries:
            citation = {
                "title": "",
                "authors": "",
                "journal": "",
                "year": None,
                "abstract": "",
                "doi": "",
                "keywords": "",
                "relevance_score": 0.5
            }
            
            # Extract fields
            title_match = re.search(r'title\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if title_match:
                citation["title"] = title_match.group(1).strip()
            
            author_match = re.search(r'author\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if author_match:
                # Convert BibTeX author format to standard
                authors = author_match.group(1).replace(' and ', '; ')
                citation["authors"] = authors
            
            journal_match = re.search(r'journal\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if journal_match:
                citation["journal"] = journal_match.group(1).strip()
            
            year_match = re.search(r'year\s*=\s*["{]?(\d{4})["}]?', entry, re.IGNORECASE)
            if year_match:
                citation["year"] = int(year_match.group(1))
            
            abstract_match = re.search(r'abstract\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if abstract_match:
                citation["abstract"] = abstract_match.group(1).strip()
            
            doi_match = re.search(r'doi\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if doi_match:
                citation["doi"] = doi_match.group(1).strip()
            
            keywords_match = re.search(r'keywords\s*=\s*["{]([^"}]+)["}]', entry, re.IGNORECASE)
            if keywords_match:
                citation["keywords"] = keywords_match.group(1).strip()
            
            citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
            
            if citation["title"]:
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def parse_zotero(content: str, format: str) -> List[Dict[str, Any]]:
        """Parse Zotero RDF or JSON format"""
        citations = []
        
        if format == 'json':
            import json
            try:
                data = json.loads(content)
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    citation = {
                        "title": item.get("title", ""),
                        "authors": "; ".join([
                            f"{author.get('family', '')}, {author.get('given', '')}"
                            for author in item.get("author", [])
                        ]),
                        "journal": item.get("container-title", ""),
                        "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                        "abstract": item.get("abstract", ""),
                        "doi": item.get("DOI", ""),
                        "keywords": "",
                        "relevance_score": 0.5
                    }
                    
                    citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
                    
                    if citation["title"]:
                        citations.append(citation)
            except:
                pass
        
        elif format == 'rdf':
            # Parse RDF format (simplified)
            try:
                root = ET.fromstring(content)
                # This is a simplified RDF parser - full implementation would be more complex
                items = root.findall('.//{http://purl.org/net/biblio#}Article')
                
                for item in items:
                    citation = {
                        "title": "",
                        "authors": "",
                        "journal": "",
                        "year": None,
                        "abstract": "",
                        "doi": "",
                        "keywords": "",
                        "relevance_score": 0.5
                    }
                    
                    # Extract fields from RDF
                    # This would need more complete implementation
                    title_elem = item.find('.//{http://purl.org/dc/elements/1.1/}title')
                    if title_elem is not None:
                        citation["title"] = title_elem.text
                    
                    if citation["title"]:
                        citations.append(citation)
            except:
                pass
        
        return citations
    
    @staticmethod
    def parse_csv(content: str) -> List[Dict[str, Any]]:
        """Parse CSV format"""
        import csv
        import io
        
        citations = []
        reader = csv.DictReader(io.StringIO(content))
        
        for row in reader:
            citation = {
                "title": row.get("Title", row.get("title", "")),
                "authors": row.get("Authors", row.get("authors", "")),
                "journal": row.get("Journal", row.get("journal", "")),
                "year": None,
                "abstract": row.get("Abstract", row.get("abstract", "")),
                "doi": row.get("DOI", row.get("doi", "")),
                "keywords": row.get("Keywords", row.get("keywords", "")),
                "relevance_score": 0.5
            }
            
            # Try to extract year
            year_str = row.get("Year", row.get("year", row.get("Publication Year", "")))
            if year_str and str(year_str).isdigit():
                citation["year"] = int(year_str)
            
            citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
            
            if citation["title"]:
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def parse_pubmed_txt(content: str) -> List[Dict[str, Any]]:
        """Parse PubMed text format"""
        citations = []
        entries = content.split('\n\n')
        
        for entry in entries:
            if not entry.strip():
                continue
            
            citation = {
                "title": "",
                "authors": "",
                "journal": "",
                "year": None,
                "abstract": "",
                "doi": "",
                "pmid": "",
                "keywords": "",
                "relevance_score": 0.5
            }
            
            lines = entry.strip().split('\n')
            
            # Simple pattern matching for PubMed text format
            for i, line in enumerate(lines):
                if i == 1 and not citation["title"]:  # Title is usually second line
                    citation["title"] = line.strip()
                elif line.startswith("PMID:"):
                    citation["pmid"] = line.replace("PMID:", "").strip()
                elif "doi:" in line.lower():
                    doi_match = re.search(r'doi:\s*([^\s]+)', line, re.IGNORECASE)
                    if doi_match:
                        citation["doi"] = doi_match.group(1)
            
            # Try to extract authors (usually third line)
            if len(lines) > 2:
                potential_authors = lines[2].strip()
                if not any(word in potential_authors.lower() for word in ['abstract', 'background', 'objective']):
                    citation["authors"] = potential_authors
            
            # Extract year from various possible locations
            for line in lines:
                year_match = re.search(r'\b(19|20)\d{2}\b', line)
                if year_match and not citation["year"]:
                    citation["year"] = int(year_match.group(0))
                    break
            
            citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
            
            if citation["title"]:
                citations.append(citation)
        
        return citations
    
    @staticmethod
    def _extract_pubmed_xml(article: ET.Element) -> Dict[str, Any]:
        """Extract citation from PubMed XML element"""
        citation = {
            "title": "",
            "authors": "",
            "journal": "",
            "year": None,
            "abstract": "",
            "doi": "",
            "pmid": "",
            "keywords": "",
            "relevance_score": 0.5
        }
        
        # Extract title
        title_elem = article.find('.//ArticleTitle')
        if title_elem is not None and title_elem.text:
            citation["title"] = title_elem.text
        
        # Extract authors
        authors = []
        for author in article.findall('.//Author'):
            last_name = author.find('LastName')
            first_name = author.find('ForeName')
            if last_name is not None and first_name is not None:
                authors.append(f"{last_name.text}, {first_name.text}")
        citation["authors"] = "; ".join(authors)
        
        # Extract journal
        journal_elem = article.find('.//Journal/Title')
        if journal_elem is not None and journal_elem.text:
            citation["journal"] = journal_elem.text
        
        # Extract year
        year_elem = article.find('.//PubDate/Year')
        if year_elem is not None and year_elem.text:
            citation["year"] = int(year_elem.text)
        
        # Extract abstract
        abstract_elem = article.find('.//Abstract/AbstractText')
        if abstract_elem is not None and abstract_elem.text:
            citation["abstract"] = abstract_elem.text
        
        # Extract PMID
        pmid_elem = article.find('.//PMID')
        if pmid_elem is not None and pmid_elem.text:
            citation["pmid"] = pmid_elem.text
        
        # Extract DOI
        for id_elem in article.findall('.//ArticleId'):
            if id_elem.get('IdType') == 'doi':
                citation["doi"] = id_elem.text
        
        # Extract keywords
        keywords = []
        for keyword in article.findall('.//Keyword'):
            if keyword.text:
                keywords.append(keyword.text)
        citation["keywords"] = "; ".join(keywords)
        
        citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
        
        return citation
    
    @staticmethod
    def _extract_generic_xml(record: ET.Element) -> Dict[str, Any]:
        """Extract citation from generic XML element"""
        citation = {
            "title": "",
            "authors": "",
            "journal": "",
            "year": None,
            "abstract": "",
            "doi": "",
            "keywords": "",
            "relevance_score": 0.5
        }
        
        # Try common field names
        title_fields = ['title', 'Title', 'article-title', 'ArticleTitle']
        for field in title_fields:
            elem = record.find(f'.//{field}')
            if elem is not None and elem.text:
                citation["title"] = elem.text
                break
        
        # Extract authors
        authors = []
        author_fields = ['author', 'Author', 'authors', 'Authors']
        for field in author_fields:
            for author_elem in record.findall(f'.//{field}'):
                if author_elem.text:
                    authors.append(author_elem.text)
        citation["authors"] = "; ".join(authors)
        
        # Extract other fields similarly...
        # This is a simplified version
        
        citation["relevance_score"] = CitationParser._calculate_relevance_score(citation)
        
        return citation
    
    @staticmethod
    def _calculate_relevance_score(citation: Dict[str, Any]) -> float:
        """Calculate relevance score based on citation completeness"""
        score = 0.0
        weights = {
            "title": 0.2,
            "authors": 0.15,
            "abstract": 0.25,
            "journal": 0.15,
            "year": 0.1,
            "doi": 0.1,
            "keywords": 0.05
        }
        
        for field, weight in weights.items():
            if citation.get(field):
                score += weight
        
        return round(score, 2)