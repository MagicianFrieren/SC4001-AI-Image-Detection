# Round 7 data protocol

This round is a data audit and split definition only. No feature extraction, training, evaluation, image export or model scoring was performed.

## Source and approved deviation

`TheKernel01/Tiny-GenImage` at revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`, all 14 SHA-256-verified train shards (28,000 source rows).
SD14 has zero images in Tiny-GenImage revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69. User approved SD15 (ID 6) as the closest substitute; this deviates from the official GenImage SD1.4 training protocol.
The earlier Step 1 stop report is retained unchanged as a historical record.

Licence reported by the user: **CC BY-NC-SA 4.0**. Underlying ImageNet and generator rights still apply. No images are redistributed.

## Exclusion and deduplication

Every original is checked against all 1,600 Round 2 images using the exact `extract_record` pixel-SHA256 definition and dHash distance <=3. Matching rows are excluded before any sampling. All remaining generators and labels are deduplicated together in ascending numeric source-row order; keep the earliest eligible representative. The deduplication is greedy, not transitive clustering. No retained pair can have equal pixel hashes or dHash distance <=3. There is no crop-hash filter.

| Stage | Rows |
|---|---:|
| eligible | 26,371 |
| old_pixel | 1,600 |
| round7_dhash_only | 24 |
| round7_pixel | 1 |
| old_dhash_only | 4 |

Counts above are exclusive removal categories, not pair counts. Full per-generator accounting is in `data_audit.json`; row-level decisions remain local under `artifacts/round7/`.

## Fixed fitting pool and three partitions

Pool/test seed: `20261031`. Partition seeds: `[20261101, 20261102, 20261103]`. Reserve test first, then select one fixed balanced Real+SD15 fitting pool. Only assignments within that same pool change between seeds.

| Partition | Real | SD15 | Total |
|---|---:|---:|---:|
| expert_fit | 750 | 750 | 1500 |
| gate_fit | 250 | 250 | 500 |
| calibration | 125 | 125 | 250 |
| selection | 125 | 125 | 250 |

Targets per class are 750/250/125/125. If necessary, scale proportionally using largest remainders; ties follow the displayed partition order. All four partitions are class-balanced and disjoint at the original-image level.

## Reserved test (identical across all three seeds)

| Generator | Role | Originals |
|---|---|---:|
| SD15 | in_domain | 250 |
| ADM | cross_generator | 500 |
| BigGAN | cross_generator | 500 |
| GLIDE | cross_generator | 500 |
| Midjourney | cross_generator | 500 |
| VQDM | cross_generator | 500 |
| Wukong | cross_generator | 500 |
| Real | shared_real | 500 |

Unique test originals: **3750**. One shared Real pool is reused across generator comparisons and counted only once in pooled metrics. No test original enters the fitting pool. Test records contain identifiers, labels and hashes only, with no image bytes, thumbnails, features or scores.

Shortfalls (target minus actual):

```json
{
  "test": {
    "SD15": 0,
    "ADM": 0,
    "BigGAN": 0,
    "GLIDE": 0,
    "Midjourney": 0,
    "VQDM": 0,
    "Wukong": 0,
    "Real": 0
  },
  "fitting_per_class": {
    "expert_fit": 0,
    "gate_fit": 0,
    "calibration": 0,
    "selection": 0
  }
}
```

## Exposure and interpretation

ADM, BigGAN and SD15 were heavily inspected in Rounds 1-6. SD15 is now the training domain and is reported only as in-domain. GLIDE, VQDM, Midjourney and Wukong were lightly exposed in a separate, independent team-member pilot; its results informed no Round 6 or Round 7 decision.
Round 7 test images were not used or inspected in rounds 1–6 of this repository. Some (116 of 3,750) also appeared in a separate, independent team-member pilot drawn from the same source; that pilot did not inform any Round 7 setting.
**Training-unseen is not the same as blind.** These tests use previously exposed generator domains and are not a strictly blind generator benchmark.

## Limitations

- The user reports an independent check found overlap of 116/3,750 reserved-test images and 94/2,500 fitting-pool images with a separate, independent team-member pilot drawn from the same Tiny-GenImage revision. That pilot is not part of this repository, its results informed no Round 6 or Round 7 decision, and it is not used or referenced as data. No external pilot files were accessed for this documentation correction.
- Training-unseen is not the same as blind; generator domains were previously exposed even though these test images were not used or inspected in rounds 1–6 of this repository.
- Real sources are JPEG and fake sources PNG with generator-specific sizes; harmonisation controls but does not remove these confounds.

## Format and size confounds

Real sources are **100% JPEG** and generated sources **100% PNG**. Real dimensions vary (500x375 is most common). BigGAN is 128x128; ADM/GLIDE/VQDM 256x256; SD15/Wukong 512x512. Midjourney has 1,999 images at 1024x1024 and one at 512x512. Complete source and selected format/size histograms are in the audit.
Harmonisation controls observable format/size differences but does not remove source compression history, effective resolution, content or generator confounds. Common128 is the inherited resize/crop control, not proof that the confounds are eliminated; no new harmonisation pipeline was introduced here.

## Settings inherited unchanged from Round 6

- Preprocessing: original (RGB centre fit to 224, bicubic) and common128 (RGB centre fit to 128, then bicubic resize to 224).
- Conditions: clean, JPEG70, JPEG50 and resize05 (224 -> 112 -> 224, bicubic; JPEG subsampling=2). Gate fitting/calibration/selection use clean/JPEG70/resize05; reserved-test condition recipe remains clean/JPEG50/resize05.
- Frozen CLIP ViT-B/32, standardized logistic heads with C=1, four fixed 56px PatchShuffle views, all four matched-dose expert recipes, raw/temperature calibration, fusion rules, gate architecture/optimization/initialization seeds, threshold and selection policies are unchanged.
- Round 6's fixed image/view counts are historical. Round 7 derives those counts from the new partitions while preserving per-original total weight one and one-third degradation doses. The three partition seeds are distinct from the unchanged three gate initialization seeds.
The protocol embeds the historical Round 6 protocol and pins all inherited implementation files by SHA-256. Do not run the old round runners for Round 7: they write their historical output paths. A future Round 7 runner must consume these assignments separately.
**Open decision:** whether to add extra test-only degradations later. No extra degradations have been approved or applied.

## Reproduction and verification

Verification must use the repository-pinned Pillow 12.2.0 environment because older Pillow changes JPEG pixel hashes. Anaconda initially had Pillow 10.4, which changed 321 historical real pixel hashes (one also changed dHash); that preliminary inventory is not used. A new isolated `artifacts/round7/pillow_12_2/` install supplies the already-pinned version without altering Anaconda or requirements. Both new scripts prefer that local install when present; otherwise use the environment's pinned version. Final preparation requires all 1,600 historical pixel/dHashes to match exactly.

Run from the repository root:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python scripts/round7_test_data.py
python -u scripts/round7_prepare_data.py
python -u scripts/round7_verify_data.py
python scripts/verify_shared_results.py
```

Preparation refuses to overwrite any result or document. Existing completed source/hash inventories are read-only inputs. To verify an already prepared round, skip preparation. The independent verifier rechecks every shard, rehashes all original images, independently replays exclusion/deduplication and seeded sampling, checks disjointness and preserved historical bytes, and performs no model operations. Its optional `--report` creates a new report exclusively.

The shared-results verifier reads already saved predictions to check historical results; it does not score any new image. Commit only the new Round 7 scripts, results and documents; keep downloaded shards and local audit caches out of Git.
