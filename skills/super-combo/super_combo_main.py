#!/usr/bin/env python3
"""
SUPER COMBO - Research Turbo + Project Brain Integration

Turkish Code Analysis + Repository Intelligence
"""

import sys
import os
from pathlib import Path

# İçeri import et
sys.path.insert(0, str(Path(__file__).parent))


class SuperCombo:
    """Research + Project Analysis birleştirme"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 SUPER COMBO - Research Turbo + Project Brain")
        print("="*80)
    
    def analyze_with_research(self, repo_path: str, research_query: str):
        """Repo'yu makale bulup analiz et
        
        Args:
            repo_path: Repository yolu
            research_query: Arama terimi (örn: "AI agents")
        """
        
        print(f"\n🔬 SUPER COMBO ANALYSIS")
        print(f"Repository: {repo_path}")
        print(f"Research Topic: {research_query}")
        print("-" * 80)
        
        try:
            from super_combo_research_turbo import ResearchTurbo
            from super_combo_project_brain import ProjectBrain
            
            # Step 1: Research
            print("\n[STEP 1] Searching for research papers...")
            turbo = ResearchTurbo()
            papers = turbo.search_papers(research_query, max_results=5)
            
            if not papers:
                print("❌ No papers found")
                return
            
            print(f"✓ Found {len(papers)} papers")
            
            # Step 2: Project Analysis
            print("\n[STEP 2] Analyzing repository structure...")
            brain = ProjectBrain(repo_path)
            
            if not brain.repo_path.exists():
                print(f"❌ Repository not found: {repo_path}")
                return
            
            brain.analyze_structure()
            architecture = brain.detect_architecture()
            risks = brain.identify_risks()
            
            # Step 3: Integration
            print("\n[STEP 3] Generating integrated report...")
            
            report = self._generate_combo_report(papers, brain, architecture)
            
            print("\n" + report)
            
            return {
                'papers': papers,
                'architecture': architecture,
                'risks': risks,
                'report': report
            }
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Make sure super_combo_*.py files are in same directory")
            return None
    
    def _generate_combo_report(self, papers, brain, architecture) -> str:
        """Combo rapor oluştur"""
        
        report = "\n" + "="*80 + "\n"
        report += "📊 SUPER COMBO ANALYSIS REPORT\n"
        report += "="*80 + "\n\n"
        
        # Papers
        report += "📚 RESEARCH PAPERS FOUND\n"
        report += "-"*80 + "\n"
        for i, paper in enumerate(papers, 1):
            report += f"{i}. {paper['title'][:60]}\n"
            report += f"   ArXiv: {paper['arxiv_id']}\n"
            report += f"   Date: {paper['published'][:10]}\n"
        
        # Architecture
        report += "\n\n🏗️  REPOSITORY ARCHITECTURE\n"
        report += "-"*80 + "\n"
        report += f"Detected Layers: {', '.join(architecture['layers'].keys())}\n"
        report += f"Patterns: {', '.join(architecture['patterns'])}\n"
        
        # Risks
        report += "\n\n⚠️  IDENTIFIED RISKS\n"
        report += "-"*80 + "\n"
        risks = brain.identify_risks()
        critical_risks = [r for r in risks if r['severity'] == 'Critical']
        high_risks = [r for r in risks if r['severity'] == 'High']
        
        report += f"Critical: {len(critical_risks)}\n"
        report += f"High: {len(high_risks)}\n"
        report += f"Total: {len(risks)}\n"
        
        if critical_risks:
            report += "\n🔴 CRITICAL ISSUES:\n"
            for risk in critical_risks[:3]:
                report += f"  - {Path(risk['file']).name}: {risk['suggestion']}\n"
        
        # Mapping
        report += "\n\n🔗 RESEARCH-IMPLEMENTATION MAPPING\n"
        report += "-"*80 + "\n"
        report += "Papers found can guide improvements to:\n"
        for layer in architecture['layers']:
            report += f"  • {layer.capitalize()} layer - Review with paper methodologies\n"
        
        # Suggestions
        report += "\n\n💡 SMART SUGGESTIONS\n"
        report += "-"*80 + "\n"
        report += "1. Cross-reference papers with current implementation\n"
        report += "2. Identify gaps in architecture vs. research\n"
        report += "3. Consider implementing paper recommendations\n"
        report += "4. Update documentation with research insights\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def demo(self):
        """Demo modu"""
        
        print("\n📖 SUPER COMBO DEMO\n")
        print("Example 1: Analyze hermes-agent with AI agent research\n")
        
        result = self.analyze_with_research(
            repo_path="/tmp/hermes-agent",
            research_query="AI agents autonomous systems"
        )
        
        return result


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        # Demo mode
        combo = SuperCombo()
        combo.demo()
    else:
        # Custom analysis
        repo_path = sys.argv[1]
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "AI agents"
        
        combo = SuperCombo()
        combo.analyze_with_research(repo_path, query)


if __name__ == "__main__":
    main()
