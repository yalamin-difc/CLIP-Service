"""
P1-7: compute_preprocessing_version is the one piece of live model
provenance that has no HTTP-level test elsewhere (it can only ever run
inside the real load_model(), which every HTTP-level test mocks out
entirely -- see tests/test_app.py's FakeProcessor/FakeModel pattern).
Tested here in isolation as the pure function it is.
"""
import unittest

import app as clip_service


class FakeImageProcessor:
    def __init__(self, config):
        self._config = config

    def to_dict(self):
        return dict(self._config)


class FakeProcessorWithImageProcessor:
    def __init__(self, config):
        self.image_processor = FakeImageProcessor(config)


BASE_CONFIG = {
    "size": {"shortest_edge": 224},
    "crop_size": {"height": 224, "width": 224},
    "resample": 3,
    "do_resize": True,
    "do_center_crop": True,
    "do_rescale": True,
    "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    # A field compute_preprocessing_version does not care about -- proves
    # the fingerprint only reflects the fields that actually affect
    # preprocessing behavior, not everything the processor happens to
    # carry.
    "processor_class": "CLIPImageProcessor",
}


class PreprocessingVersionTests(unittest.TestCase):
    def test_deterministic_for_identical_config(self):
        first = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(BASE_CONFIG))
        second = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(dict(BASE_CONFIG)))
        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_changes_when_the_real_preprocessing_pipeline_changes(self):
        # A different resize target is a real behavioral difference in
        # preprocessing -- the fingerprint must reflect that, not stay
        # coincidentally identical.
        changed_config = dict(BASE_CONFIG, size={"shortest_edge": 336})
        original = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(BASE_CONFIG))
        changed = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(changed_config))
        self.assertNotEqual(original, changed)

    def test_changes_when_normalization_mean_or_std_changes(self):
        changed_config = dict(BASE_CONFIG, image_mean=[0.5, 0.5, 0.5])
        original = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(BASE_CONFIG))
        changed = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(changed_config))
        self.assertNotEqual(original, changed)

    def test_ignores_fields_irrelevant_to_actual_preprocessing_behavior(self):
        with_extra = dict(BASE_CONFIG, some_unrelated_metadata="anything")
        original = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(BASE_CONFIG))
        with_extra_version = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor(with_extra))
        self.assertEqual(original, with_extra_version)

    def test_returns_none_rather_than_a_placeholder_when_no_relevant_config_exists(self):
        # Never invents a version string when the processor genuinely
        # carries none of the fields this fingerprint is built from --
        # None means "not yet knowable," not "unknown-but-treat-as-fine."
        result = clip_service.compute_preprocessing_version(FakeProcessorWithImageProcessor({}))
        self.assertIsNone(result)

    def test_falls_back_to_the_processor_itself_when_it_has_no_image_processor_attribute(self):
        # CLIPProcessor normally exposes .image_processor, but this stays
        # resilient to a processor shape that puts the config directly on
        # itself instead.
        processor = FakeImageProcessor(BASE_CONFIG)
        result = clip_service.compute_preprocessing_version(processor)
        self.assertIsNotNone(result)


class RuntimeProvenanceTests(unittest.TestCase):
    def tearDown(self):
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None

    def test_reflects_current_live_state_not_hardcoded_constants(self):
        clip_service.embedding_dimension = 512
        clip_service.preprocessing_version = "abc123"
        provenance = clip_service.runtime_provenance()
        self.assertEqual(provenance["modelId"], clip_service.MODEL_NAME)
        self.assertEqual(provenance["modelRevision"], clip_service.MODEL_REVISION)
        self.assertEqual(provenance["embeddingDimension"], 512)
        self.assertEqual(provenance["preprocessingVersion"], "abc123")
        self.assertEqual(provenance["scoringVersion"], clip_service.SCORING_VERSION)
        self.assertEqual(provenance["serviceVersion"], clip_service.SERVICE_VERSION)

    def test_reflects_a_changed_live_dimension_immediately(self):
        # Proves this is read live at call time, not cached/snapshotted --
        # a real re-embedding that produces a different dimension (e.g. a
        # hot-swapped model) is reflected on the very next call.
        clip_service.embedding_dimension = 512
        first = clip_service.runtime_provenance()
        clip_service.embedding_dimension = 768
        second = clip_service.runtime_provenance()
        self.assertEqual(first["embeddingDimension"], 512)
        self.assertEqual(second["embeddingDimension"], 768)


if __name__ == "__main__":
    unittest.main()
