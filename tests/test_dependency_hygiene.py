import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEAD_MODULES = ["storage.py", "match_service.py"]

# P2-10: these were only ever used by the dead modules above (or, for the
# git+ CLIP source, superseded entirely by app.py's transformers-based
# CLIP loading) -- never by app.py or anything it imports.
REMOVED_REQUIREMENTS = [
    "torchvision",
    "torchaudio",
    "sentence-transformers",
    "scikit-learn",
    "ftfy",
    "openai/CLIP.git",
]


class DeadModuleRemovalTests(unittest.TestCase):
    def test_duplicate_storage_and_match_service_modules_are_gone(self):
        for name in DEAD_MODULES:
            self.assertFalse(
                (REPO_ROOT / name).exists(),
                f"{name} should have been removed as a dead/duplicate module",
            )


class RequirementsHygieneTests(unittest.TestCase):
    def test_unused_ml_stack_removed_from_requirements(self):
        for filename in ["requirements.txt", "requirements-gpu.txt"]:
            content = (REPO_ROOT / filename).read_text()
            for dep in REMOVED_REQUIREMENTS:
                self.assertNotIn(
                    dep,
                    content,
                    f"{dep} should not appear in {filename}; it is unused "
                    "by app.py and its real dependencies",
                )

    def test_transformers_direct_dependency_still_pinned(self):
        # transformers is what app.py actually uses for CLIP inference, and
        # itself requires regex/tqdm -- those stay pinned directly too.
        content = (REPO_ROOT / "requirements.txt").read_text()
        for dep in ["transformers", "torch", "regex", "tqdm"]:
            self.assertIn(dep, content)


if __name__ == "__main__":
    unittest.main()
