import unittest
from unittest.mock import patch

import numpy as np

import app as clip_service


class FakeTokenizer:
    def __init__(self, model_max_length):
        self.model_max_length = model_max_length


class RecordingProcessor:
    """Records the kwargs it was called with so tests can assert the
    truncation contract without needing a real CLIPTokenizer installed."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.calls = []

    def __call__(self, images=None, text=None, **kwargs):
        if text is not None:
            self.calls.append({"text": text, **kwargs})
            # A minimal but shape-correct stand-in for a real tokenizer's
            # text_features output -- length/content-independent, since
            # this test is about the truncation kwargs reaching the
            # processor, not about a specific tokenizer's own behavior.
            return {"text_features": np.asarray([[1.0, 2.0, 3.0]])}
        raise AssertionError("missing text input")


class FakeModel:
    def get_text_features(self, **inputs):
        return inputs["text_features"]


class TextMaxTokensTests(unittest.TestCase):
    def test_prefers_the_tokenizer_own_configured_value(self):
        processor = RecordingProcessor(tokenizer=FakeTokenizer(model_max_length=128))
        self.assertEqual(clip_service.text_max_tokens(processor), 128)

    def test_falls_back_to_77_when_no_tokenizer_is_present(self):
        processor = RecordingProcessor(tokenizer=None)
        self.assertEqual(clip_service.text_max_tokens(processor), clip_service.CLIP_TEXT_CONTEXT_LENGTH)
        self.assertEqual(clip_service.CLIP_TEXT_CONTEXT_LENGTH, 77)

    def test_falls_back_to_77_when_configured_value_is_an_unset_sentinel(self):
        # HF tokenizers sometimes leave model_max_length at a huge sentinel
        # (e.g. 1e30) when no real limit was recorded in the checkpoint.
        processor = RecordingProcessor(tokenizer=FakeTokenizer(model_max_length=int(1e30)))
        self.assertEqual(clip_service.text_max_tokens(processor), 77)

    def test_falls_back_to_77_for_an_implausibly_small_configured_value(self):
        processor = RecordingProcessor(tokenizer=FakeTokenizer(model_max_length=1))
        self.assertEqual(clip_service.text_max_tokens(processor), 77)


class TextEmbeddingTruncationTests(unittest.TestCase):
    """P12 found explicit safe truncation missing from the text encoding
    path; these pin that text_embedding_for now always asks the processor
    to truncate at the tokenizer's own context length, for any input --
    short/long, English/Arabic alike -- so a long description can never
    fail the whole request merely for exceeding the token limit."""

    def setUp(self):
        self.tokenizer = FakeTokenizer(model_max_length=77)
        self.processor = RecordingProcessor(tokenizer=self.tokenizer)
        self.model_patch = patch.object(
            clip_service, "load_model", return_value=(FakeModel(), self.processor)
        )
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def _assert_safe_call(self, text):
        embedding = clip_service.text_embedding_for(text)
        self.assertEqual(len(embedding), 3)
        self.assertEqual(len(self.processor.calls), 1)
        call = self.processor.calls[0]
        self.assertTrue(call["truncation"])
        self.assertEqual(call["max_length"], 77)
        self.assertEqual(call["text"], [text])

    def test_short_english_description(self):
        self._assert_safe_call("Blue backpack left near gate 4")

    def test_long_english_description_does_not_fail_for_exceeding_the_token_limit(self):
        long_text = " ".join(["black leather wallet with initials embossed on the front"] * 40)
        self._assert_safe_call(long_text)

    def test_short_arabic_description(self):
        self._assert_safe_call("حقيبة سوداء قرب البوابة الرئيسية")

    def test_long_arabic_description_does_not_fail_for_exceeding_the_token_limit(self):
        long_text = " ".join(["محفظة جلدية سوداء بجانب المدخل الرئيسي للمهرجان"] * 40)
        self._assert_safe_call(long_text)

    def test_uses_the_tokenizer_configured_context_length_when_available(self):
        self.tokenizer.model_max_length = 128
        embedding = clip_service.text_embedding_for("short text")
        self.assertEqual(len(embedding), 3)
        self.assertEqual(self.processor.calls[0]["max_length"], 128)


if __name__ == "__main__":
    unittest.main()
