# Rock photogrammetry

[Aerial Rocks 02](https://polyhaven.com/a/aerial_rocks_02), by Rob Tuytel, from Poly Haven. The source page publishes this asset under [CC0](https://polyhaven.com/license). The diffuse, OpenGL normal, displacement and roughness maps are the 2K JPG distribution. Their measured material scale is 50 metres per tile.

Powered by Poly Haven. `sources.json` records the official download URLs, original MD5 checksums and local SHA-256 checksums. The maps are local build inputs, packed into generated Cycles `.blend` files. They are photographic rock textures, not substituted water or a rendered background photograph.

The runtime ridge uses a neutral mineral variant of the photographed color map, generated reproducibly by `export-alpine-runtime.py`: 15% source chroma, 85% luminance, and 0.8 intensity. Normals and roughness remain original. This derived texture is packed in the native GLB and sampled by the common FVX baker.
