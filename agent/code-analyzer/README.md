# Code Analyzer Agent

Türkçe Kod Analiz Ajanı - Python kodlarını otomatik olarak analiz eder, sorunları bulur ve kalite puanı verir.

## Özellikler

✅ **Söz Dizimi Kontrolü** - Hatalı söz dizimini tespit eder
✅ **Kod Kalitesi Analizi** - Kod yapısını değerlendirir
✅ **Güvenlik Taraması** - SQL injection, eval(), exec() gibi riskli kodları bulur
✅ **Best Practices Kontrolü** - Modern Python standartlarını kontrol eder
✅ **Puanlama Sistemi** - 0-100 arası kalite puanı verir
✅ **Detaylı Rapor** - Sorunları ve önerileri listeler

## Kurulum

```bash
# Dosyaları klonla
cd hermes-agent/agent/code-analyzer

# Python 3.8+ gerekli
python3 --version
```

## Kullanım

### Basit Kullanım

```bash
python3 analyzer.py <dosya_yolu>
```

Örnek:
```bash
python3 analyzer.py ~/my_script.py
```

### Test Modu

```bash
python3 analyzer.py
```

Bu, dahili test kodunu çalıştırır ve sonuçları gösterir.

## Çıktı Örneği

```
============================================================
KOD ANALİZİ BAŞLANIYOR: ~/example.py
============================================================

✓ Söz dizimi: TAMAM
✓ Kod kalitesi: Kontrol edildi
✗ GÜVENLİK: 2 sorun bulundu
✓ Best practices: Kontrol edildi

============================================================
ANALİZ RAPORU
============================================================

Kod Kalitesi Puanı: 68/100

⚠️  SORUNLAR (3):
  1. 1 boş fonksiyon var
  2. SQL injection riski var (string birleştirme)
  3. eval() kullanıyor - güvenlik riski!

💡 ÖNERİLER (2):
  1. Yeterli yorum/açıklama ekle
  2. Type hints ekle (modern Python)

============================================================
```

## Kontrol Edilen Sorunlar

### Güvenlik
- SQL injection riskli kodlar
- eval() ve exec() kullanımı
- pickle ile güvenlik riskler
- Parametresiz veritabanı sorguları

### Kod Kalitesi
- Söz dizimi hataları
- Boş fonksiyonlar
- Yetersiz yorumlar
- Çok uzun fonksiyonlar (30+ satır)

### Best Practices
- Global değişken kullanımı
- Type hints eksikliği
- Genel Exception kullanımı
- Docstring eksikliği

## Puanlama

- **90-100**: Mükemmel - Hiç sorun yok
- **75-89**: İyi - Az sorun
- **50-74**: Orta - Bazı sorunlar var
- **0-49**: Kötü - Ciddi sorunlar

## Entegrasyon

### Hermes Gateway ile

```python
from analyzer import CodeAnalyzer

analyzer = CodeAnalyzer()
analyzer.analyze("path/to/code.py")
```

### CI/CD Pipeline'da

```bash
#!/bin/bash
python3 analyzer.py $1
if [ $? -eq 0 ]; then
    exit 0
else
    exit 1
fi
```

## Örnekler

### Örnek 1: Hatalı Kod

```python
# bad_code.py
def fetch_data(input):
    query = "SELECT * FROM users WHERE id = " + input
    eval(query)
    
try:
    pass
except Exception:
    pass
```

Çalıştırma:
```bash
python3 analyzer.py bad_code.py
```

**Sonuç**: 63/100 puan, 4 sorun bulundu

### Örnek 2: İyi Kod

```python
# good_code.py
from typing import Optional, List

def fetch_user(user_id: int) -> Optional[dict]:
    """
    Kullanıcı bilgisini veritabanından al
    """
    try:
        # Parametreli sorgu (SQL injection'a karşı güvenli)
        query = "SELECT * FROM users WHERE id = ?"
        result = execute_query(query, (user_id,))
        return result
    except DatabaseError as e:
        logger.error(f"DB error: {e}")
        return None
```

**Sonuç**: 92/100 puan, 0 kritik sorun

## Geliştirme

Yeni kontroller eklemek:

```python
class CodeAnalyzer:
    def _check_custom_rule(self, code):
        """Özel kural ekle"""
        if "pattern" in code:
            self.issues.append("Yapılacak kontrol")
            self.score -= 5
```

## Katkı

PR göndermek hoş geldiniz! Lütfen:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/my-feature`)
3. Değişiklikleri commit edin (`git commit -am 'Add feature'`)
4. Branch'e push edin (`git push origin feature/my-feature`)
5. Pull Request açın

## Lisans

MIT

## Yazar

Created for Hermes Agent Framework
Turkish Code Analysis Agent
