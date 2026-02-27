#!/usr/bin/env python3
"""
Research Turbo - arXiv makale bulup PDF'den özet çıkaran

SUPER COMBO Part 1/3
"""

import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional


class ResearchTurbo:
    """arXiv makale analiz sistemi"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query?"
        self.papers = []
    
    def search_papers(self, query: str, max_results: int = 5) -> List[Dict]:
        """arXiv'den makale ara
        
        Args:
            query: Arama terimi (örn: "AI agents")
            max_results: Kaç makale
            
        Returns:
            Makale listesi
        """
        import urllib.parse
        
        print(f"\n📚 SEARCHING ARXIV: {query}")
        print("-" * 60)
        
        try:
            # arXiv API query (URL encoded)
            encoded_query = urllib.parse.quote(query)
            search_query = f'search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
            url = self.base_url + search_query
            
            response = urllib.request.urlopen(url)
            root = ET.fromstring(response.read())
            
            papers = []
            
            # XML parse et
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = {
                    'title': entry.find('{http://www.w3.org/2005/Atom}title').text.strip(),
                    'authors': [
                        author.find('{http://www.w3.org/2005/Atom}name').text 
                        for author in entry.findall('{http://www.w3.org/2005/Atom}author')
                    ],
                    'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text.strip(),
                    'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
                    'arxiv_id': entry.find('{http://www.w3.org/2005/Atom}id').text.split('/abs/')[-1],
                    'pdf_url': entry.find('{http://www.w3.org/2005/Atom}id').text.replace('abs', 'pdf') + '.pdf',
                }
                papers.append(paper)
                
                print(f"✓ {paper['title'][:50]}...")
            
            self.papers = papers
            return papers
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def extract_methodology(self, paper: Dict) -> Dict:
        """Makaleden metodoloji çıkar"""
        summary = paper['summary']
        
        methodology = {
            'paper_title': paper['title'],
            'arxiv_id': paper['arxiv_id'],
            'key_methods': [],
            'algorithms': [],
            'frameworks': [],
        }
        
        # Basit anahtar kelime taraması
        keywords = {
            'key_methods': ['approach', 'method', 'algorithm', 'framework', 'architecture'],
            'algorithms': ['neural network', 'transformer', 'attention', 'gan', 'reinforcement'],
            'frameworks': ['pytorch', 'tensorflow', 'huggingface', 'jax'],
        }
        
        text = summary.lower()
        for key, words in keywords.items():
            for word in words:
                if word in text:
                    methodology[key].append(word.title())
        
        return methodology
    
    def generate_comparison_table(self, papers: List[Dict]) -> str:
        """Makale karşılaştırma tablosu oluştur"""
        
        table = "\n📊 PAPERS COMPARISON TABLE\n"
        table += "=" * 100 + "\n"
        table += f"{'Title':<40} {'Authors':<20} {'Date':<12} {'Key Methods':<20}\n"
        table += "-" * 100 + "\n"
        
        for paper in papers:
            title = paper['title'][:37] + "..." if len(paper['title']) > 40 else paper['title']
            authors = ", ".join(paper['authors'][:1]) if paper['authors'] else "N/A"
            authors = authors[:17] + "..." if len(authors) > 20 else authors
            date = paper['published'][:10]
            
            methodology = self.extract_methodology(paper)
            methods = ", ".join(methodology['algorithms'][:2]) if methodology['algorithms'] else "N/A"
            methods = methods[:17] + "..." if len(methods) > 20 else methods
            
            table += f"{title:<40} {authors:<20} {date:<12} {methods:<20}\n"
        
        table += "=" * 100 + "\n"
        return table
    
    def generate_summary(self) -> str:
        """Tüm makalelerin özeti"""
        
        summary = "\n🔍 RESEARCH SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for i, paper in enumerate(self.papers, 1):
            summary += f"{i}. {paper['title']}\n"
            summary += f"   ArXiv: {paper['arxiv_id']}\n"
            summary += f"   Authors: {', '.join(paper['authors'][:3])}\n"
            summary += f"   Date: {paper['published'][:10]}\n"
            summary += f"   Summary: {paper['summary'][:200]}...\n"
            summary += f"   PDF: {paper['pdf_url']}\n\n"
        
        summary += "=" * 80 + "\n"
        return summary


if __name__ == "__main__":
    turbo = ResearchTurbo()
    
    # Test: AI Agents araştırması
    papers = turbo.search_papers("AI agents autonomous", max_results=5)
    
    if papers:
        print(turbo.generate_comparison_table(papers))
        print(turbo.generate_summary())
    else:
        print("❌ No papers found")
