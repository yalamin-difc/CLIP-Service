"""P14.1: branding/content refinement only -- CLIP matching, SigLIP2
matching, embedding dimensions, model configuration, authentication, demo
facade security, tenant/site isolation, and rate limiting are all
untouched (see tests/test_v2_api.py, tests/test_engines.py,
tests/test_demo_facade.py for that coverage, all still passing
unmodified). This file covers the 14 rebrand-specific checks from the
P14.1 brief.
"""
import pathlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as clip_service
import internal_auth
from test_app import FakeModel, FakeProcessor, TEST_SECRET

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class RebrandTests(unittest.TestCase):
    def setUp(self):
        self.auth_patch = patch.object(internal_auth, "INTERNAL_JWT_SECRET", TEST_SECRET)
        self.auth_patch.start()
        self.model_patch = patch.object(clip_service, "load_model", return_value=(FakeModel(), FakeProcessor()))
        self.model_patch.start()
        self.dimension_patch = patch.object(clip_service, "EXPECTED_EMBEDDING_DIMENSION", 3)
        self.dimension_patch.start()
        clip_service.embedding_dimension = 3
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())
        clip_service.rate_limit_windows.clear()
        self.client = TestClient(clip_service.app)
        self.response = self.client.get("/")
        self.html = self.response.text
        self.js = (REPO_ROOT / "static" / "clip-console.js").read_text(encoding="utf-8")
        self.css = (REPO_ROOT / "static" / "clip-console.css").read_text(encoding="utf-8")

    def tearDown(self):
        self.dimension_patch.stop()
        self.model_patch.stop()
        self.auth_patch.stop()
        clip_service.embedding_dimension = None
        clip_service.preprocessing_version = None
        clip_service.set_item_repository(clip_service.InMemoryItemRepository())

    # 1: visible site name contains "Multimodal AL Engine".
    def test_1_visible_site_name(self):
        self.assertIn("Multimodal AL Engine", self.html)

    # 2: browser title contains "Multimodal AL Engine — Urban Intelligence".
    def test_2_browser_title(self):
        self.assertIn("<title>Multimodal AL Engine — Urban Intelligence</title>", self.html)

    # 3: visible UI no longer displays "AL-AMEN TECHNOLOGY" (or any casing
    # variant of the old company branding) anywhere in the rendered page
    # or its static assets.
    def test_3_no_al_amen_branding_anywhere_visible(self):
        for haystack, label in ((self.html, "rendered HTML"), (self.js, "clip-console.js"), (self.css, "clip-console.css")):
            self.assertNotIn("AL-AMEN", haystack.upper(), f"AL-AMEN branding found in {label}")
            self.assertNotIn("Al-Amen", haystack, f"Al-Amen branding found in {label}")

    def test_3b_no_old_product_name_clip_ai_engine(self):
        for haystack, label in ((self.html, "rendered HTML"), (self.js, "clip-console.js"), (self.css, "clip-console.css")):
            self.assertNotIn("CLIP AI Engine", haystack, f"old 'CLIP AI Engine' branding found in {label}")

    # 4: hero contains "Multimodal Asset Recovery AI".
    def test_4_hero_headline(self):
        self.assertIn("Multimodal Asset Recovery AI", self.html)

    # 5: CLIP model reference present.
    def test_5_clip_model_reference(self):
        self.assertIn("openai/clip-vit-base-patch32", self.html)

    # 6: SigLIP2 model reference present.
    def test_6_siglip2_model_reference(self):
        self.assertIn("google/siglip2-so400m-patch14-384", self.html)

    # 7: 512-D is associated with CLIP (same model card / kv block).
    def test_7_512d_associated_with_clip(self):
        clip_card_start = self.html.index('id="model-card-clip"')
        clip_card_end = self.html.index("</div>\n        <div class=\"card model-card\" id=\"model-card-siglip2\"")
        clip_card_html = self.html[clip_card_start:clip_card_end]
        self.assertIn("openai/clip-vit-base-patch32", clip_card_html)
        self.assertIn("512-D", clip_card_html)

    # 8: 1152-D is associated with SigLIP2 (same model card / kv block).
    def test_8_1152d_associated_with_siglip2(self):
        siglip2_card_start = self.html.index('id="model-card-siglip2"')
        siglip2_card_html = self.html[siglip2_card_start : siglip2_card_start + 1500]
        self.assertIn("google/siglip2-so400m-patch14-384", siglip2_card_html)
        self.assertIn("1152-D", siglip2_card_html)

    # 9: "Experimental / Uncalibrated" remains visible for SigLIP2.
    def test_9_siglip2_experimental_uncalibrated_visible(self):
        self.assertIn("Experimental / Uncalibrated", self.html)

    # 10: CLIP / SigLIP2 / Compare live-demo functionality is unchanged --
    # spot-checked here (full behavioral coverage lives in
    # tests/test_demo_facade.py, run as part of the full suite).
    def test_10_mode_tabs_present_and_unchanged_ids(self):
        for mode_id in ("mode-tab-clip", "mode-tab-siglip2", "mode-tab-compare"):
            self.assertIn(f'id="{mode_id}"', self.html)
        # The endpoints themselves are untouched technical identifiers.
        self.assertIn("/demo/match", self.js)

    # 11: Arabic translation exists for the changed strings.
    def test_11_arabic_translations_present(self):
        self.assertIn("محرك الذكاء متعدد الوسائط", self.js)  # Multimodal AL Engine
        self.assertIn("الذكاء الاصطناعي متعدد الوسائط لاستعادة الأصول", self.js)  # hero
        self.assertIn("SigLIP2", self.js)  # Arabic explanation still names both models by their English names
        self.assertIn("CLIP", self.js)

    # 12: RTL remains functional.
    def test_12_rtl_still_functional(self):
        self.assertIn('setAttribute("dir", "rtl")', self.js)
        self.assertIn('data-lang="ar"', self.html)

    # 13: no JWT or secret exposed to JavaScript (full coverage in
    # test_festival_console.py; spot-checked here against the rebranded page).
    def test_13_no_jwt_or_secret_exposed(self):
        self.assertNotIn(TEST_SECRET, self.html)
        self.assertNotIn(TEST_SECRET, self.js)
        self.assertNotIn("Authorization", self.js)
        for forbidden in ("sign_internal_token", "HS256", "hmac", "jsonwebtoken"):
            self.assertNotIn(forbidden, self.js)

    # 14: model IDs stay LTR even though they sit inside an Arabic-capable
    # page -- the model-card rows wrap each model ID in dir="ltr".
    def test_14_model_ids_kept_ltr(self):
        clip_id_index = self.html.index("openai/clip-vit-base-patch32")
        preceding = self.html[max(0, clip_id_index - 120) : clip_id_index]
        self.assertIn('dir="ltr"', preceding)
        siglip2_id_index = self.html.index("google/siglip2-so400m-patch14-384")
        preceding = self.html[max(0, siglip2_id_index - 120) : siglip2_id_index]
        self.assertIn('dir="ltr"', preceding)

    # Never imply CLIP and SigLIP2 share one embedding space, and never
    # declare a "winner" between them.
    def test_no_shared_embedding_space_claim(self):
        self.assertNotIn("winner", self.html.lower())
        self.assertIn("separate", self.html.lower())

    # Section 6: hero secondary line shows both dimensions and roles
    # without declaring either engine a winner.
    def test_hero_secondary_model_meta_line(self):
        hero_start = self.html.index('class="hero"')
        hero_html = self.html[hero_start : hero_start + 3000]
        self.assertIn("512-D", hero_html)
        self.assertIn("1152-D", hero_html)
        self.assertIn("Production Baseline", hero_html)
        self.assertIn("Experimental / Uncalibrated", hero_html)

    # Section 4: new description explains both models without implying a
    # combined embedding space.
    def test_new_description_mentions_both_models_and_separate_spaces(self):
        self.assertIn("CLIP", self.js)
        self.assertIn("SigLIP2", self.js)
        self.assertIn("separate", self.js.lower())
        # The historical single-CLIP paragraph text must be gone.
        self.assertNotIn("CLIP is one intelligence layer", self.html)


if __name__ == "__main__":
    unittest.main()
