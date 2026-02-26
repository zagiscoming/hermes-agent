#!/usr/bin/env python3
"""
Test Suite for Code Analyzer Agent
"""

import sys
import tempfile
from pathlib import Path
from analyzer import CodeAnalyzer


def test_syntax_check():
    """Test söz dizimi kontrolü"""
    print("\n[TEST 1] Söz dizimi hatası kontrolü...")
    
    bad_code = """
def foo(
    pass
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(bad_code)
        f.flush()
        
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(f.name)
        
        if len(analyzer.issues) > 0 and "Söz dizimi" in str(analyzer.issues):
            print("✅ PASS - Söz dizimi hatası tespit edildi")
            return True
        else:
            print("❌ FAIL - Söz dizimi hatası bulunamadı")
            return False


def test_security_check():
    """Test güvenlik taraması"""
    print("\n[TEST 2] Güvenlik taraması...")
    
    risky_code = """
def unsafe_query(user_id):
    sql = "SELECT * FROM users WHERE id = " + user_id
    eval(sql)
    return sql
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(risky_code)
        f.flush()
        
        analyzer = CodeAnalyzer()
        analyzer.analyze(f.name)
        
        has_sql_issue = any("SQL" in issue for issue in analyzer.issues)
        has_eval_issue = any("eval" in issue for issue in analyzer.issues)
        
        if has_sql_issue and has_eval_issue:
            print("✅ PASS - Güvenlik sorunları tespit edildi")
            return True
        else:
            print("❌ FAIL - Güvenlik sorunları bulunamadı")
            return False


def test_quality_check():
    """Test kod kalitesi kontrolü"""
    print("\n[TEST 3] Kod kalitesi kontrolü...")
    
    poor_quality = """
def foo():
    pass

def bar():
    pass
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(poor_quality)
        f.flush()
        
        analyzer = CodeAnalyzer()
        analyzer.analyze(f.name)
        
        has_empty_func = any("boş fonksiyon" in issue for issue in analyzer.issues)
        
        if has_empty_func:
            print("✅ PASS - Boş fonksiyonlar tespit edildi")
            return True
        else:
            print("❌ FAIL - Boş fonksiyonlar bulunamadı")
            return False


def test_best_practices():
    """Test best practices kontrolü"""
    print("\n[TEST 4] Best practices kontrolü...")
    
    poor_practices = """
def calculate():
    try:
        x = 1 + 2
    except Exception:
        pass
    return x
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(poor_practices)
        f.flush()
        
        analyzer = CodeAnalyzer()
        analyzer.analyze(f.name)
        
        has_exception = any("Exception" in issue for issue in analyzer.issues)
        
        if has_exception:
            print("✅ PASS - Best practices ihlali tespit edildi")
            return True
        else:
            print("❌ FAIL - Best practices ihlali bulunamadı")
            return False


def test_good_code():
    """Test iyi kod örneği"""
    print("\n[TEST 5] İyi kod örneği...")
    
    good_code = '''
"""İyi yazılmış kod modülü"""

from typing import Optional
import logging


def calculate_sum(a: int, b: int) -> int:
    """
    İki sayıyı topla
    
    Args:
        a: İlk sayı
        b: İkinci sayı
        
    Returns:
        Toplam
    """
    return a + b


def safe_query(user_id: int) -> Optional[dict]:
    """Güvenli SQL sorgusu"""
    try:
        query = "SELECT * FROM users WHERE id = ?"
        result = execute_query(query, (user_id,))
        return result
    except DatabaseError as e:
        logging.error(f"DB Error: {e}")
        return None
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(good_code)
        f.flush()
        
        analyzer = CodeAnalyzer()
        analyzer.analyze(f.name)
        
        # İyi kod minimal sorun içermeli
        if analyzer.score >= 70:
            print("✅ PASS - İyi kod yüksek puan aldı")
            return True
        else:
            print(f"❌ FAIL - İyi kod düşük puan aldı: {analyzer.score}")
            return False


def run_all_tests():
    """Tüm testleri çalıştır"""
    print("\n" + "="*60)
    print("KOD ANALİZ AGENT - TEST SERESİ")
    print("="*60)
    
    tests = [
        test_syntax_check,
        test_security_check,
        test_quality_check,
        test_best_practices,
        test_good_code,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*60)
    print("TEST SONUÇLARI")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Başarılı: {passed}/{total}")
    
    if all(results):
        print("\n✅ TÜM TESTLER BAŞARILI!")
        return 0
    else:
        print("\n❌ BAZI TESTLER BAŞARISIZ")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
