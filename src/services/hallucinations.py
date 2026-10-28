"""
What Whisper writes that nobody said: stock lines over silence, noise or
music, and runs of the same words. Removed before a transcript is returned.
"""
import re

_INVENTED = re.compile(
    r'^(terima ?kasih( (banyak|ya|telah menonton|sudah menonton|atas perhatiannya))*'
    r'|sampai jumpa( lagi)?( di video selanjutnya)?|selamat menonton|jangan lupa (like|subscribe).*'
    r'|subtitle.*|disponsori oleh.*|(yeay|yey|wow|ah|eh|oh|uh|hm+|hmm+|hai|halo)( (yeay|yey|wow|ah|eh|oh|uh|hm+|hmm+|hai|halo))*)$'
)
# A run of the same 1-4 words, e.g. "udah udah udah udah ...".
_LOOP = re.compile(r'\b((?:\S+\s+){0,3}?\S+)(?:\s+\1\b){3,}', re.IGNORECASE)


def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()


def is_invented(text: str) -> bool:
    return bool(_INVENTED.match(normalize(text)))


def collapse_loops(text: str) -> str:
    return _LOOP.sub(lambda m: f'{m.group(1)} {m.group(1)} …', text)
