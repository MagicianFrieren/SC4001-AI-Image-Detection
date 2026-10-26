"""Fast synthetic tests for Round 7 data safety; no models or real test images."""
import sys
sys.dont_write_bytecode = True

import copy
import hashlib
import io
import itertools
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import round7_prepare_data as prep
import round7_verify_data as verify
import prepare_data as historical
from PIL import Image


def record(index, generator="SD15", h=None, pixel=None):
    digest = hashlib.sha256(str(index).encode()).hexdigest()
    return {"source_id": f"train:{index}", "source_row": index, "generator": generator,
            "label": int(generator != "Real"), "pixel_sha256": pixel or digest,
            "dhash": f"{int(digest[:16], 16) if h is None else h:016x}",
            "source_bytes_sha256": hashlib.sha256((str(index) + "bytes").encode()).hexdigest()}


def fixture():
    rows = [record(i, "SD15") for i in range(1399)]
    rows += [record(i + 2000, "Real") for i in range(2000)]
    for i, name in enumerate(["ADM", "BigGAN", "GLIDE", "Midjourney", "VQDM", "Wukong"]):
        rows += [record(10000 + i * 100 + j, name) for j in range(10)]
    return rows


class Round7Tests(unittest.TestCase):
    def test_hash_definition_matches_historical_png_and_exif_jpeg(self):
        names = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]
        rng = random.Random(17)
        image = Image.frombytes("RGB", (37, 29), bytes(rng.randrange(256) for _ in range(37 * 29 * 3)))
        for fmt in ("PNG", "JPEG"):
            stream = io.BytesIO()
            exif = Image.Exif()
            exif[274] = 6
            image.save(stream, format=fmt, exif=exif)
            row = {"image": {"bytes": stream.getvalue()}, "generator": 6, "label": 1}
            expected, _ = historical.extract_record(row, 123)
            actual = prep.record_image(row, 123, names)
            for key in ("pixel_sha256", "dhash", "width", "height", "label", "generator", "source_id"):
                self.assertEqual(actual[key], expected[key])

    def test_radius_three_search_has_no_block_boundary_false_negatives(self):
        index = prep.NearIndex()
        original = 0x918AC1436D82F901
        index.add(original, "original")
        for radius in range(4):
            for bits in itertools.combinations([0, 15, 16, 31, 32, 47, 48, 63], radius):
                query = original
                for bit in bits:
                    query ^= 1 << bit
                self.assertEqual(index.match(query), "original")
        self.assertIsNone(index.match(original ^ sum(1 << b for b in [0, 16, 32, 48])))

    def test_exclusion_priority_and_global_deduplication(self):
        old = [record(999, "Real", h=0, pixel="old")]
        rows = [record(0, h=2**64 - 1, pixel="old"), record(1, h=7),
                record(2, "Real", h=65535, pixel="kept"), record(3, h=65534),
                record(4, h=65535 << 16, pixel="kept"), record(5, h=2**32 - 1)]
        kept, decisions = prep.filter_records(list(reversed(rows)), old)
        self.assertEqual([r["source_row"] for r in kept], [2, 5])
        self.assertEqual([d["reason"] for d in decisions],
                         ["old_pixel", "old_dhash_only", "eligible", "round7_dhash_only", "round7_pixel", "eligible"])
        independent = verify.independently_filter(rows, old, decisions)[0]
        self.assertEqual(independent, kept)

    def test_proportional_scaling_and_tie_break(self):
        self.assertEqual(list(prep.scaled_counts(1250).values()), [750, 250, 125, 125])
        self.assertEqual(list(prep.scaled_counts(10).values()), [6, 2, 1, 1])
        self.assertEqual(list(prep.scaled_counts(5).values()), [3, 1, 1, 0])
        for budget in range(1251):
            self.assertEqual(sum(prep.scaled_counts(budget).values()), budget)

    def test_reserved_first_fixed_pool_balanced_seeds_and_shortfall(self):
        pool, test, assignments, counts, tests = prep.define_splits(fixture())
        self.assertEqual(tests["SD15"], 250)
        self.assertEqual(tests["Real"], 500)
        self.assertEqual(sum(counts.values()), 1149)
        self.assertEqual(prep.define_splits(list(reversed(fixture()))), (pool, test, assignments, counts, tests))
        self.assertFalse({r["source_id"] for r in pool} & {r["source_id"] for r in test})
        expected_ids = {r["source_id"] for r in pool}
        lookup = {r["source_id"]: r for r in pool}
        for assignment in assignments:
            flattened = sum(assignment["partitions"].values(), [])
            self.assertEqual(set(flattened), expected_ids)
            self.assertEqual(len(flattened), len(expected_ids))
            for part, ids in assignment["partitions"].items():
                self.assertEqual(sum(lookup[i]["label"] for i in ids), counts[part])
                self.assertEqual(len(ids), 2 * counts[part])
        self.assertNotEqual(assignments[0]["partitions"], assignments[1]["partitions"])

    def test_independent_split_verifier_detects_tampering(self):
        prep.ART.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_", dir=prep.ART) as temp:
            out = Path(temp)
            rows = fixture()
            pool, test, assignments, counts, tests = prep.define_splits(rows)
            pool = [{k: r[k] for k in prep.CONTENT_KEYS} for r in pool]
            test = [{k: r[k] for k in prep.CONTENT_KEYS} for r in test]
            prep.write_new(out / "fitting_pool_manifest.jsonl", pool, lines=True)
            prep.write_new(out / "reserved_test_manifest.jsonl", test, lines=True)
            for assignment in assignments:
                assignment.update(fitting_pool_sha256=prep.sha(out / "fitting_pool_manifest.jsonl"),
                                  reserved_test_sha256=prep.sha(out / "reserved_test_manifest.jsonl"))
            protocol = {"counts_per_class_each_seed": counts, "reserved_test_counts": tests,
                        "shortfalls": {"test": {g: prep.TEST_TARGETS[g] - tests[g] for g in tests},
                                       "fitting_per_class": {p: t - counts[p] for p, t in zip(prep.PARTS, prep.TARGETS)}}}
            with patch.object(verify, "OUT", out):
                verify.verify_splits(rows, pool, test, assignments, protocol)
                bad = copy.deepcopy(assignments)
                bad[0]["partitions"]["gate_fit"][0] = bad[0]["partitions"]["expert_fit"][0]
                with self.assertRaises(ValueError):
                    verify.verify_splits(rows, pool, test, bad, protocol)
                bad_test = copy.deepcopy(test)
                bad_test[0]["pixel_sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    verify.verify_splits(rows, pool, bad_test, assignments, protocol)

    def test_no_output_overwrite_and_early_refusal(self):
        prep.ART.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_", dir=prep.ART) as temp:
            path = Path(temp) / "existing.json"
            prep.write_new(path, {"keep": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                prep.write_new(path, {"keep": False})
            with patch.object(prep, "OUTPUTS", [path]):
                with self.assertRaises(FileExistsError):
                    prep.main()
            self.assertEqual(path.read_bytes(), before)

    def test_empty_fitting_budget_refused(self):
        rows = [record(i, "SD15") for i in range(250)] + [record(i + 1000, "Real") for i in range(1000)]
        with self.assertRaises(ValueError):
            prep.define_splits(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
