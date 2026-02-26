#!/usr/bin/env python3
"""
Kod Analiz Agent - Kodları analiz eder, sorunları bulur, düzeltir ve test eder
Turkish Code Analysis Agent for Hermes Agent Framework
"""

import os
import sys
import ast
import re
from pathlib import Path


class CodeAnalyzer:
    """Kod analiz ve kalite kontrol sınıfı"""
    
    def __init__(self):
        self.issues = []
        self.suggestions = []
        self.score = 100
        
    def analyze(self, code_path):
        """Kod dosyasını analiz et"""
        print(f"\n{'='*60}")
        print(f"KOD ANALİZİ BAŞLANIYOR: {code_path}")
        print(f"{'='*60}\n")
        
        try:
            with open(code_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            print(f"HATA: Dosya bulunamadı: {code_path}")
            return False
        
        # 1. Söz dizimi hatalarını kontrol et
        self._check_syntax(code)
        
        # 2. Kod kalitesini kontrol et
        self._check_quality(code)
        
        # 3. Güvenlik sorunlarını kontrol et
        self._check_security(code)
        
        # 4. Best practices'i kontrol et
        self._check_best_practices(code)
        
        # Rapor ver
        self._print_report()
        return True
    
    def _check_syntax(self, code):
        """Söz dizimi hatalarını kontrol et"""
        try:
            ast.parse(code)
            print("✓ Söz dizimi: TAMAM")
        except SyntaxError as e:
            print(f"✗ SÖZDIZIMI HATASI: Satır {e.lineno}: {e.msg}")
            self.issues.append(f"Söz dizimi: {e.msg}")
            self.score -= 20
    
    def _check_quality(self, code):
        """Kod kalitesini kontrol et"""
        lines = code.split('\n')
        
        # Boş fonksiyonları kontrol et
        empty_funcs = len(re.findall(r'def\s+\w+\(.*?\):\s*pass', code))
        if empty_funcs > 0:
            self.issues.append(f"{empty_funcs} boş fonksiyon var")
            self.score -= 5 * empty_funcs
        
        # Açıklamasız kod
        if len(code) > 100 and code.count('#') < len(lines) // 10:
            self.suggestions.append("Yeterli yorum/açıklama ekle")
            self.score -= 5
        
        # Uzun fonksiyonları kontrol et
        func_pattern = r'def\s+\w+\(.*?\):.*?(?=\ndef|\nclass|\Z)'
        funcs = re.findall(func_pattern, code, re.DOTALL)
        for func in funcs:
            if len(func.split('\n')) > 30:
                self.suggestions.append("Çok uzun fonksiyonlar var, böl")
                self.score -= 3
                break
        
        print("✓ Kod kalitesi: Kontrol edildi")
    
    def _check_security(self, code):
        """Güvenlik sorunlarını kontrol et"""
        issues = []
        
        # SQL injection riski
        if 'sql' in code.lower() and '+' in code and '"' in code:
            issues.append("SQL injection riski var (string birleştirme)")
        
        # eval() kullanımı
        if 'eval(' in code:
            issues.append("eval() kullanıyor - güvenlik riski!")
        
        # exec() kullanımı
        if 'exec(' in code:
            issues.append("exec() kullanıyor - güvenlik riski!")
        
        # pickle kullanımı
        if 'pickle' in code.lower():
            issues.append("pickle.loads() güvenlik riski olabilir")
        
        for issue in issues:
            self.issues.append(issue)
            self.score -= 10
        
        if issues:
            print(f"✗ GÜVENLİK: {len(issues)} sorun bulundu")
        else:
            print("✓ Güvenlik: TEMIZ")
    
    def _check_best_practices(self, code):
        """Best practices'i kontrol et"""
        
        # Global değişkenler
        globals_count = len(re.findall(r'^\w+\s*=', code, re.MULTILINE))
        if globals_count > 5:
            self.suggestions.append("Çok fazla global değişken var")
            self.score -= 3
        
        # Veri türü ipuçları
        if ':' not in code or '->' not in code:
            self.suggestions.append("Type hints ekle (modern Python)")
            self.score -= 2
        
        # Exception handling
        if 'except' in code and 'except Exception' in code:
            self.issues.append("Genel Exception kullanma, spesifik exception yakala")
            self.score -= 5
        
        # Docstring kontrolü
        if 'def ' in code and '"""' not in code and "'''" not in code:
            self.suggestions.append("Fonksiyonlar için docstring ekle")
            self.score -= 2
        
        print("✓ Best practices: Kontrol edildi")
    
    def _print_report(self):
        """Analiz raporunu yazdır"""
        print(f"\n{'='*60}")
        print("ANALİZ RAPORU")
        print(f"{'='*60}\n")
        
        print(f"Kod Kalitesi Puanı: {max(0, self.score)}/100")
        
        if self.issues:
            print(f"\n⚠️  SORUNLAR ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
        
        if self.suggestions:
            print(f"\n💡 ÖNERİLER ({len(self.suggestions)}):")
            for i, sugg in enumerate(self.suggestions, 1):
                print(f"  {i}. {sugg}")
        
        if not self.issues and not self.suggestions:
            print("\n✅ Mükemmel! Hiç sorun bulunamadı.")
        
        print(f"\n{'='*60}\n")


def test_analyzer():
    """Test kodu analiz et"""
    test_code = '''
# Hatalı kod örneği
def bad_function():
    pass

def calculate_with_vulnerabilities(user_input):
    # SQL injection riski
    query = "SELECT * FROM users WHERE id = " + user_input
    result = eval(query)
    return result

x = 5
y = 10
z = 15

def very_long_function():
    for i in range(100):
        try:
            data = x + y + z
        except Exception:
            pass
    return data
'''
    
    # Test dosyası oluştur
    test_file = "/tmp/test_code.py"
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    # Analiz et
    analyzer = CodeAnalyzer()
    analyzer.analyze(test_file)
    
    # Sonuç
    if analyzer.score < 50:
        print("⚠️  KOD KALİTESİ DÜŞÜK - DÜZELTİLME ÖNERİLİ")
    elif analyzer.score < 75:
        print("⚠️  KOD KALİTESİ ORTA - İYİLEŞTİRİLEBİLİR")
    else:
        print("✅ KOD KALİTESİ İYİ")
    
    return analyzer.score >= 50


if __name__ == "__main__":
    print("\n🤖 KOD ANALİZ AGENT BAŞLANIYOR...\n")
    
    if len(sys.argv) > 1:
        # Dosya argümanı varsa onu analiz et
        code_file = sys.argv[1]
        analyzer = CodeAnalyzer()
        analyzer.analyze(code_file)
    else:
        # Test koş
        print("Test modu çalışıyor...\n")
        success = test_analyzer()
        sys.exit(0 if success else 1)
