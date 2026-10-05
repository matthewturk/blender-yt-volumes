# yt_idv patches

cycles-yt needs yt_idv features that are still open pull requests upstream.
`scripts/setup_yt_idv.sh` reproduces the exact yt_idv this repo is tested
against: upstream `main` at `780f233` plus
`yt_idv-780f233-integration.patch`, checked out into `./yt_idv`, where
`uv sync --group ytidv` installs it from.

The integration patch is those PRs merged onto that commit:

| PR | What cycles-yt uses it for |
|---|---|
| [#284](https://github.com/yt-project/yt_idv/pull/284) order-independent iteration | faster draws in order-independent modes |
| [#285](https://github.com/yt-project/yt_idv/pull/285) bindless textures | one draw call per frame; big scenes render ~100x faster |
| [#286](https://github.com/yt-project/yt_idv/pull/286) matrix and pixel order | correct non-square readback |
| [#287](https://github.com/yt-project/yt_idv/pull/287) scene transforms | subset scaling, deep zooms, shot space (required) |
| [#288](https://github.com/yt-project/yt_idv/pull/288) transfer function conversion | the yt Transfer Function mode |

plus one conflict resolution and two fixes:

- **Merge of #287 with #284/#285**, in `block_collection.py`:
  `viewpoint_order()` passes `camera.position` through `_tree_viewpoint()`,
  so draw order stays right after a rescale. The other conflicts are
  additive.
- **`0001-Draw-zero-cell-kd-tree-bricks-as-empty-instead-of-fa.patch`**
  (applies to upstream `main`): yt's kd-tree makes bricks with no cells along
  an axis on deep hierarchies (9 of DeeplyNestedZoom's 7,540, in stock yt
  4.4.2 too), which made `glTexStorage3D` fail. They're drawn empty.
- **`0002-tf_conversion-floor-tf_min-at-float32-tiny-not-eps.patch`**
  (applies to #288): `tf_min` was floored at float32 eps, which hid every
  value below 1e-7 of the field's maximum; fields spanning more than 7
  decades rendered mostly black.

The two standalone patches are the same changes as in the integration patch,
in `git am` form for upstreaming. As the PRs merge, the integration patch
shrinks; once they all have, `main` plus these two fixes (if not merged
themselves) is enough.

yt is the PyPI release; a yt with `amr_kdtools.viewpoint_node_ids` (from the
`kdtree_walk_speedup` work) makes bindless draws faster still, and yt_idv
falls back without it.
