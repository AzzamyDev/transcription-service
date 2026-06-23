"""
Sentiment analysis and suspicious phrase detection for transcripts.
Keyword-based, no external dependencies.
"""
from typing import List


_NEGATIVE_KEYWORDS = [
    'penipuan', 'penipu', 'menipu', 'ditipu', 'tipu',
    'ancaman', 'mengancam', 'diancam', 'ancam',
    'marah', 'kemarahan', 'emosi', 'murka', 'memaki', 'kesal',
    'curiga', 'mencurigakan', 'dicurigai',
    'paksaan', 'memaksa', 'dipaksa', 'paksa',
    'bahaya', 'berbahaya', 'mengancam',
    'korupsi', 'suap', 'sogok', 'menyuap',
    'narkoba', 'sabu', 'ganja', 'narkotika',
    'kekerasan', 'menyerang', 'serang',
    'teror', 'terorisme', 'intimidasi', 'radik', 'jihad',
    'penculikan', 'menculik', 'diculik',
    'kabur', 'melarikan diri', 'buronan',
    'bom', 'senjata', 'membunuh', 'pembunuhan',
]

_POSITIVE_KEYWORDS = [
    'rindu', 'merindukan',
    'sehat', 'kesehatan',
    'baik', 'baik-baik',
    'senang', 'gembira', 'bahagia', 'kebahagiaan',
    'doa', 'mendoakan', 'berdoa',
    'syukur', 'bersyukur', 'alhamdulillah',
    'terima kasih', 'makasih',
    'sukses', 'berhasil', 'semangat',
    'harmonis', 'damai', 'aman',
]

# (keyword, priority) — lower number = higher priority
_FLAGGED_HIGH: List[str] = [
    'transfer rekening', 'nomor rekening', 'kode otp', 'kode verifikasi',
    'kode password', 'kirim uang', 'bayar dulu', 'jangan bilang siapa-siapa',
    'sabu', 'narkoba', 'ganja', 'kabur', 'lompat', 'senjata api', 'bom',
    'serang', 'korupsi', 'suap', 'sogok', 'teror', 'jihad', 'radik',
]

_FLAGGED_MEDIUM: List[str] = [
    'hadiah', 'menang', 'pemenang', 'pulsa', 'token', 'rekening',
    'transfer', 'barang titipan', 'paket', 'ambil dulu', 'jemput',
    'jual', 'harga',
]


def analyze_sentiment(summary: str) -> str:
    """
    Classify sentiment of an AI-generated summary as POSITIVE, NEGATIVE, or NEUTRAL.
    Returns NEUTRAL when summary is empty or unavailable.
    """
    if not summary or not summary.strip():
        return 'NEUTRAL'

    text = summary.lower()

    neg_hits = sum(1 for kw in _NEGATIVE_KEYWORDS if kw in text)
    pos_hits = sum(1 for kw in _POSITIVE_KEYWORDS if kw in text)

    if neg_hits > pos_hits:
        return 'NEGATIVE'
    if pos_hits > neg_hits:
        return 'POSITIVE'
    return 'NEUTRAL'


def extract_flagged_phrases(segments: list, max_phrases: int = 10) -> List[str]:
    """
    Scan transcript segments and return phrases containing suspicious keywords.
    HIGH-priority matches come first. Each phrase includes the speaker label.

    Args:
        segments: List of dicts with 'speaker' and 'text' keys.
        max_phrases: Maximum number of phrases to return.
    """
    if not segments:
        return []

    high_matches: List[str] = []
    medium_matches: List[str] = []
    seen: set = set()

    for seg in segments:
        speaker = seg.get('speaker', '')
        text = seg.get('text', '')
        phrase = f"{speaker}: {text}".strip()
        text_lower = text.lower()

        if phrase in seen:
            continue

        matched_high = any(kw in text_lower for kw in _FLAGGED_HIGH)
        if matched_high:
            high_matches.append(phrase)
            seen.add(phrase)
            continue

        matched_medium = any(kw in text_lower for kw in _FLAGGED_MEDIUM)
        if matched_medium:
            medium_matches.append(phrase)
            seen.add(phrase)

    combined = high_matches + medium_matches
    return combined[:max_phrases]
