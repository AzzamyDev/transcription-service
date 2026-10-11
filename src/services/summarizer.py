"""
AI Summarization - Ollama (Self-hosted, Gratis)
Model: qwen2.5:14b — terbaik untuk Bahasa Indonesia
"""
from typing import Optional, Dict
import time
import requests
import json

from src.core.config import Config
from src.core.exceptions import SummarizationError


# Default Ollama endpoint
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"

# Re-check Ollama availability every 5 minutes so a temporary outage
# at worker startup doesn't permanently disable summarization.
_AVAILABILITY_TTL = 300
# Every request must ask for the same context, or Ollama reloads the model.
# 8k fits a transcript chunk and keeps VRAM free for the other models.
CONTEXT_TOKENS = 8192


class AISummarizer:
    """Generate summaries via Ollama - self-hosted, gratis"""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or Config.OLLAMA_BASE_URL if hasattr(Config, 'OLLAMA_BASE_URL') else OLLAMA_BASE_URL
        self.model = model or Config.OLLAMA_MODEL if hasattr(Config, 'OLLAMA_MODEL') else OLLAMA_MODEL
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0

    def is_available(self) -> bool:
        """Cek apakah Ollama bisa diakses dan model tersedia (re-check tiap 5 menit)"""
        now = time.time()
        if self._available is not None and (now - self._available_checked_at) < _AVAILABILITY_TTL:
            return self._available

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self._available = self.model.split(":")[0] in model_names
                if not self._available:
                    print(f"[Summarizer] Model '{self.model}' tidak ditemukan di Ollama. "
                          f"Jalankan: ollama pull {self.model}", flush=True)
            else:
                self._available = False
        except Exception as e:
            print(f"[Summarizer] Ollama tidak bisa diakses di {self.base_url}: {e}", flush=True)
            self._available = False

        self._available_checked_at = now
        return self._available

    def last_known_status(self) -> str:
        """For /health: the last check, without calling Ollama."""
        if self._available is None:
            return 'unknown'
        return 'available' if self._available else 'unavailable'

    def generate_json(self, prompt: str, schema: dict, max_tokens: int = 1500) -> dict:
        """A reply constrained to a JSON schema, parsed"""
        text = self._generate(prompt, max_tokens=max_tokens, schema=schema)
        try:
            return json.loads(text)
        except ValueError:
            raise SummarizationError(f"Ollama returned invalid JSON: {text[:200]}")

    def _generate(self, prompt: str, max_tokens: int = 800, schema: Optional[dict] = None) -> str:
        """Kirim prompt ke Ollama dan ambil hasilnya"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                # Calls arrive minutes apart; reloading 9 GB each time is slow.
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": max_tokens,
                    "num_ctx": CONTEXT_TOKENS,
                    "top_p": 0.9,
                }
            }
            if schema is not None:
                payload["format"] = schema

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # model lokal bisa lebih lambat
            )

            if response.status_code != 200:
                raise SummarizationError(f"Ollama error {response.status_code}: {response.text}")

            result = response.json()
            return result.get("response", "").strip()

        except requests.exceptions.Timeout:
            raise SummarizationError("Ollama timeout - coba lagi atau gunakan model yang lebih kecil")
        except requests.exceptions.ConnectionError:
            raise SummarizationError(f"Tidak bisa konek ke Ollama di {self.base_url}")
        except Exception as e:
            raise SummarizationError(f"Gagal generate summary: {str(e)}")

    def create_summary(
        self,
        transcript: str,
        statistics: Dict[str, any],
        language: str = 'id'
    ) -> Optional[str]:
        """Buat ringkasan dari transkrip"""
        if not self.is_available():
            return None

        if language == 'id':
            prompt = self._create_indonesian_prompt(transcript, statistics)
        else:
            prompt = self._create_english_prompt(transcript, statistics)

        return self._generate(prompt, max_tokens=800)

    @staticmethod
    def _create_indonesian_prompt(transcript: str, statistics: Dict) -> str:
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        return f"""Buatlah ringkasan SINGKAT dan PADAT dari transkrip berikut dalam bahasa Indonesia.

STATISTIK:
- Durasi: {statistics['total_duration']/60:.1f} menit
- Pembicara: {statistics['num_speakers']}
- Total kata: {statistics['total_words']:,}

TRANSKRIP:
{transcript}

INSTRUKSI:
- Maksimal 5-7 kalimat
- Fokus pada poin utama saja
- Langsung to the point
- Gunakan bullet points jika perlu
- Hindari penjelasan berlebihan
- Balas HANYA dengan ringkasan, tanpa penjelasan tambahan

Format:
- Topik utama (1 kalimat)
- Poin-poin penting (3-5 bullet points)
- Kesimpulan (1 kalimat jika ada)

RINGKASAN:"""

    @staticmethod
    def _create_english_prompt(transcript: str, statistics: Dict) -> str:
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        return f"""Create a SHORT and CONCISE summary of this transcript in English.

STATISTICS:
- Duration: {statistics['total_duration']/60:.1f} minutes
- Speakers: {statistics['num_speakers']}
- Total words: {statistics['total_words']:,}

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Maximum 5-7 sentences
- Focus on main points only
- Straight to the point
- Use bullet points if needed
- Avoid over-explanation
- Reply ONLY with the summary, no extra commentary

Format:
- Main topic (1 sentence)
- Key points (3-5 bullet points)
- Conclusion (1 sentence if applicable)

SUMMARY:"""

    def create_meeting_summary(
        self,
        transcript: str,
        statistics: Dict[str, any],
        meeting_type: str = 'general',
        language: str = 'id'
    ) -> Optional[str]:
        """Buat ringkasan khusus meeting"""
        if not self.is_available():
            return None

        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."

        prompts = {
            'standup': self._standup_prompt(transcript, statistics, language),
            'interview': self._interview_prompt(transcript, statistics, language),
            'general': self._general_meeting_prompt(transcript, statistics, language)
        }

        prompt = prompts.get(meeting_type, prompts['general'])
        return self._generate(prompt, max_tokens=600)

    @staticmethod
    def _standup_prompt(transcript: str, statistics: Dict, language: str) -> str:
        if language == 'id':
            return f"""Ringkas standup meeting ini SINGKAT. Balas HANYA dengan ringkasan.

{transcript}

Format (maksimal 3-4 kalimat per pembicara):
- Yang dikerjakan kemarin
- Yang akan dikerjakan hari ini
- Blocker (jika ada)

RINGKASAN:"""
        else:
            return f"""Summarize this standup meeting BRIEFLY. Reply ONLY with the summary.

{transcript}

Format (max 3-4 sentences per speaker):
- Done yesterday
- Will do today
- Blockers (if any)

SUMMARY:"""

    @staticmethod
    def _interview_prompt(transcript: str, statistics: Dict, language: str) -> str:
        if language == 'id':
            return f"""Ringkas interview ini SINGKAT (maksimal 6 kalimat). Balas HANYA dengan ringkasan.

{transcript}

Fokus pada:
- Pertanyaan utama
- Jawaban kunci
- Penilaian singkat

RINGKASAN:"""
        else:
            return f"""Summarize this interview BRIEFLY (max 6 sentences). Reply ONLY with the summary.

{transcript}

Focus on:
- Main questions
- Key answers
- Brief assessment

SUMMARY:"""

    @staticmethod
    def _general_meeting_prompt(transcript: str, statistics: Dict, language: str) -> str:
        if language == 'id':
            return f"""Ringkas meeting ini SINGKAT (maksimal 5-6 kalimat). Balas HANYA dengan ringkasan.

{transcript}

Sertakan:
- Topik utama (1 kalimat)
- Keputusan penting (2-3 poin)
- Action items (jika ada)

RINGKASAN:"""
        else:
            return f"""Summarize this meeting BRIEFLY (max 5-6 sentences). Reply ONLY with the summary.

{transcript}

Include:
- Main topic (1 sentence)
- Key decisions (2-3 points)
- Action items (if any)

SUMMARY:"""