"""A summary that cannot be made is reported, never fails the transcript."""
import pytest

from src.core.exceptions import SummarizationError
from src.utils.pipeline import AudioTranscriptionPipeline


class FakeSummarizer:
    def __init__(self, available=True, result='ringkasan', error=None):
        self.available, self.result, self.error = available, result, error

    def is_available(self):
        return self.available

    def create_summary(self, transcript, statistics, language='id'):
        if self.error:
            raise self.error
        return self.result


def pipeline(summarizer):
    pipe = AudioTranscriptionPipeline.__new__(AudioTranscriptionPipeline)
    pipe.summarizer = summarizer
    pipe.language = 'id'
    return pipe


@pytest.mark.parametrize('summarizer, text, include, expected', [
    (FakeSummarizer(), 'A: halo', True, ('ringkasan', 'ok')),
    (FakeSummarizer(), 'A: halo', False, (None, 'skipped')),
    (FakeSummarizer(), '  ', True, (None, 'empty')),
    (FakeSummarizer(available=False), 'A: halo', True, (None, 'unavailable')),
    (None, 'A: halo', True, (None, 'unavailable')),
    (FakeSummarizer(error=SummarizationError('timeout')), 'A: halo', True, (None, 'failed')),
    (FakeSummarizer(result=''), 'A: halo', True, (None, 'failed')),
])
def test_summary_status(summarizer, text, include, expected):
    assert pipeline(summarizer)._summarize(text, {}, include) == expected
