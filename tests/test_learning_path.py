from __future__ import annotations

import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEARNING_PATH = ROOT / "docs" / "learning-path.md"


class LearningPathTests(unittest.TestCase):
    def test_learning_path_is_reachable_from_the_two_entry_points(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        starter = (ROOT / "STARTER_GUIDE.md").read_text(encoding="utf-8")

        self.assertIn("docs/learning-path.md", readme)
        self.assertIn("docs/learning-path.md", starter)

    def test_learning_path_local_links_resolve(self) -> None:
        self.assertTrue(LEARNING_PATH.is_file())
        text = LEARNING_PATH.read_text(encoding="utf-8")
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)

        for target in links:
            target = target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local_target = target.split("#", 1)[0]
            if not local_target:
                continue
            self.assertTrue(
                (LEARNING_PATH.parent / local_target).is_file()
                if not local_target.startswith("/")
                else False,
                f"learning-path local link does not resolve: {target}",
            )

    def test_learning_path_keeps_authority_and_evidence_boundaries_explicit(self) -> None:
        text = LEARNING_PATH.read_text(encoding="utf-8").lower()

        for phrase in (
            "in-process admission",
            "governed wrapper",
            "organization authority",
            "fail-closed",
            "observed result",
            "promotion evidence",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
