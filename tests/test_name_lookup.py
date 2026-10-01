import json
import os
import tempfile
import unittest
from unittest.mock import patch

from name_normalization import fighter_name_slug, normalize_fighter_name
from name_resolver import FighterNameResolver
from predict_service import PredictionService


class _StubPredictor:
    def predict(self, vector):
        return {
            "consensus_win_probability_a": 0.64,
            "final_prediction": {
                "winner_side": "A",
                "method": "Decision",
                "win_probability": 0.64,
            },
        }


class FighterNameNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_path = os.path.join(self.temp_dir.name, "fighter_state.json")
        states = {
            "jiri-id": {
                "name": "Jiri Prochazka",
                "experience": 20,
                "elo": 1740.0,
                "glicko_rating": 1710.0,
                "glicko_rd": 70.0,
                "stance": "Orthodox",
            },
            "ankalaev-id": {
                "name": "Magomed Ankalaev",
                "experience": 22,
                "elo": 1760.0,
                "glicko_rating": 1730.0,
                "glicko_rd": 65.0,
                "stance": "Southpaw",
            },
        }
        with open(self.state_path, "w") as handle:
            json.dump(states, handle)
        self.service = PredictionService(state_file=self.state_path)
        self.service._predictor = _StubPredictor()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalizer_strips_accents_and_collapses_whitespace(self):
        self.assertEqual(
            normalize_fighter_name("  JIŘÍ   PROCHÁZKA  "),
            "jiri prochazka",
        )
        self.assertEqual(
            fighter_name_slug("Jiří", "", "Procházka"),
            "jiri-prochazka",
        )

    def test_prediction_accepts_accented_and_ascii_spellings(self):
        for spelling in ("Jiří Procházka", "Jiri Prochazka"):
            with self.subTest(spelling=spelling):
                result = self.service.predict(
                    spelling,
                    "Magomed Ankalaev",
                    with_odds=False,
                )
                self.assertNotIn("error", result)
                self.assertEqual(result["fighter_a"], "Jiri Prochazka")
                self.assertEqual(
                    result["final_prediction"]["winner"],
                    "Jiri Prochazka",
                )

    def test_existing_ascii_lookup_and_accented_search_still_work(self):
        self.assertEqual(
            self.service._resolve("Magomed Ankalaev")["matched"],
            "Magomed Ankalaev",
        )
        results = self.service.search("Jiří Procházka")
        self.assertEqual(results[0]["name"], "Jiri Prochazka")

    def test_ufc_profile_resolver_uses_normalized_slug(self):
        resolver = FighterNameResolver()

        def page_exists(first, middle, last):
            return resolver.build_slug(first, middle, last) == "jiri-prochazka"

        with patch.object(resolver, "page_exists", side_effect=page_exists):
            resolved = resolver.resolve("Jiří", "", "Procházka")

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["first"], "jiri")
        self.assertEqual(resolved["last"], "prochazka")


if __name__ == "__main__":
    unittest.main()
