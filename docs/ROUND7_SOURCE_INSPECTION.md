# Round 7 source inspection (Step 1)

Source: `TheKernel01/Tiny-GenImage`; pinned revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`.

All 14 train shards verified against Hugging Face LFS SHA-256; 28,000 images.

| ClassLabel | Generator (source spelling) | Images | Encoded formats | Most common encoded sizes |
|---|---|---:|---|---|
| 0 | Real | 14000 | JPEG: 14000 | 500x375: 3280, 500x333: 1313, 375x500: 986, 333x500: 445, 500x334: 339 |
| 1 | ADM | 2000 | PNG: 2000 | 256x256: 2000 |
| 2 | BigGAN | 2000 | PNG: 2000 | 128x128: 2000 |
| 3 | GLIDE | 2000 | PNG: 2000 | 256x256: 2000 |
| 4 | Midjourney | 2000 | PNG: 2000 | 1024x1024: 1999, 512x512: 1 |
| 5 | SD14 | 0 | none | none |
| 6 | SD15 | 2000 | PNG: 2000 | 512x512: 2000 |
| 7 | VQDM | 2000 | PNG: 2000 | 256x256: 2000 |
| 8 | Wukong | 2000 | PNG: 2000 | 512x512: 2000 |

The complete size histogram and each verified shard hash are in `results/round7/source_inventory.json`.
Sizes are image-header dimensions before EXIF orientation or preprocessing. Image headers were inspected without exporting images or computing model scores.

**STOP: SD1.4 has no images in the complete pinned training source.**
Per the requested stop condition, no exclusions, fitting pool, seed assignments, reserved test set, or final data protocol have been created. Steps 2–5 and their verifier remain pending a revised source or training-generator decision.
