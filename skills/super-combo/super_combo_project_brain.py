#!/usr/bin/env python3
"""
Project Brain - Repo yapı + mimari analiz

SUPER COMBO Part 2/3
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict


class ProjectBrain:
    """Repository analiz sistemi"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.structure = {}
        self.files = []
        self.dependencies = []
        self.risk_areas = []
    
    def analyze_structure(self) -> Dict:
        """Repo yapısını analiz et"""
        
        print(f"\n🧠 ANALYZING REPO: {self.repo_path}")
        print("-" * 60)
        
        structure = {}
        
        if not self.repo_path.exists():
            print(f"❌ Path not found: {self.repo_path}")
            return structure
        
        # Dizin ağacı oluştur
        for root, dirs, files in os.walk(self.repo_path):
            # Gizli ve cache klasörleri atla
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            level = root.replace(str(self.repo_path), '').count(os.sep)
            relative = os.path.basename(root)
            
            if relative == os.path.basename(self.repo_path):
                current = structure
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                self.files.append({
                    'name': file,
                    'path': os.path.join(root, file),
                    'size': os.path.getsize(os.path.join(root, file))
                })
        
        self.structure = structure
        print(f"✓ Scanned {len(self.files)} files")
        return structure
    
    def detect_architecture(self) -> Dict:
        """Mimariyi tespit et"""
        
        print("\n🏗️  DETECTING ARCHITECTURE...")
        
        architecture = {
            'layers': {},
            'patterns': [],
            'components': []
        }
        
        # Klasör isimleri mimari bulgu
        layer_keywords = {
            'frontend': ['ui', 'web', 'client', 'frontend', 'views'],
            'backend': ['api', 'server', 'backend', 'routes', 'handlers'],
            'database': ['db', 'database', 'models', 'schema', 'migrations'],
            'utils': ['utils', 'helpers', 'lib', 'tools', 'common'],
            'tests': ['test', 'tests', 'spec', 'specs'],
        }
        
        detected_layers = set()
        
        for file_info in self.files:
            path = file_info['path'].lower()
            
            for layer, keywords in layer_keywords.items():
                if any(keyword in path for keyword in keywords):
                    detected_layers.add(layer)
        
        architecture['layers'] = {layer: True for layer in detected_layers}
        
        # Pattern tespiti
        if 'database' in detected_layers and 'backend' in detected_layers:
            architecture['patterns'].append('MVC or Layered Architecture')
        
        if 'tests' in detected_layers:
            architecture['patterns'].append('Test-Driven Development')
        
        if 'frontend' in detected_layers and 'backend' in detected_layers:
            architecture['patterns'].append('Monolithic with Separation of Concerns')
        
        print(f"✓ Architecture: {', '.join(architecture['patterns'])}")
        return architecture
    
    def identify_risks(self) -> List[Dict]:
        """Risk alanlarını tespit et"""
        
        print("\n⚠️  IDENTIFYING RISKS...")
        
        risks = []
        
        # Risk indikatörleri
        for file_info in self.files:
            path = file_info['path']
            name = file_info['name'].lower()
            
            # Dosya boyutu kontrolü
            if file_info['size'] > 500000:  # 500KB+
                risks.append({
                    'file': path,
                    'type': 'Large File',
                    'severity': 'Medium',
                    'suggestion': 'Consider refactoring into smaller modules'
                })
            
            # Riskli anahtar kelimeler
            if any(x in name for x in ['temp', 'old', 'backup', 'unused', 'deprecated']):
                risks.append({
                    'file': path,
                    'type': 'Deprecated Code',
                    'severity': 'Low',
                    'suggestion': 'Remove or document why it\'s kept'
                })
            
            # SQL/DB riskler
            if any(x in name for x in ['sql', 'query', 'database']):
                risks.append({
                    'file': path,
                    'type': 'Database Logic',
                    'severity': 'High',
                    'suggestion': 'Verify parameterized queries to prevent injection'
                })
            
            # Auth riskler
            if 'auth' in name or 'security' in name or 'secret' in name:
                risks.append({
                    'file': path,
                    'type': 'Security Critical',
                    'severity': 'Critical',
                    'suggestion': 'Ensure proper secret management and encryption'
                })
        
        self.risk_areas = risks
        print(f"✓ Found {len(risks)} potential issues")
        return risks
    
    def generate_report(self) -> str:
        """Mimari rapor oluştur"""
        
        report = "\n📋 PROJECT ANALYSIS REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"📁 Repository: {self.repo_path}\n"
        report += f"📊 Total Files: {len(self.files)}\n\n"
        
        # Mimari
        architecture = self.detect_architecture()
        report += "🏗️  ARCHITECTURE\n"
        report += "-" * 80 + "\n"
        report += f"Layers: {', '.join(architecture['layers'].keys())}\n"
        report += f"Patterns: {', '.join(architecture['patterns'])}\n\n"
        
        # Risk alanları
        report += "⚠️  RISK AREAS\n"
        report += "-" * 80 + "\n"
        
        if self.risk_areas:
            for risk in self.risk_areas[:5]:  # İlk 5
                report += f"[{risk['severity']}] {risk['file']}\n"
                report += f"         → {risk['suggestion']}\n"
        else:
            report += "✅ No critical risks found\n"
        
        report += "\n" + "=" * 80 + "\n"
        return report
    
    def generate_summary(self) -> str:
        """Proje özeti"""
        
        summary = "\n✨ PROJECT SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        # Dosya türleri
        extensions = defaultdict(int)
        for file_info in self.files:
            ext = Path(file_info['name']).suffix or 'no-ext'
            extensions[ext] += 1
        
        summary += "📊 File Types:\n"
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:5]:
            summary += f"  {ext}: {count} files\n"
        
        summary += "\n🎯 Key Directories:\n"
        dirs = set()
        for file_info in self.files:
            dir_path = Path(file_info['path']).parent.relative_to(self.repo_path)
            if len(dir_path.parts) > 0:
                dirs.add(str(dir_path.parts[0]))
        
        for dir_name in sorted(dirs)[:5]:
            summary += f"  /{dir_name}/\n"
        
        summary += "\n" + "=" * 80 + "\n"
        return summary


if __name__ == "__main__":
    brain = ProjectBrain("/tmp/hermes-agent")
    
    if brain.repo_path.exists():
        brain.analyze_structure()
        arch = brain.detect_architecture()
        risks = brain.identify_risks()
        
        print(brain.generate_report())
        print(brain.generate_summary())
    else:
        print("Repository path not found")
