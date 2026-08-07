import bpy
import pytest

from cycles_volume_override import compositor_build as cb


@pytest.fixture
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield bpy.context.scene


def test_ensure_compositor_tree_raises_clear_error_on_blender_5_api(clean_scene):
    # This documents/pins the known gap: on Blender >= 5.0 (which this test
    # suite runs under, via pip bpy==5.2.0), the classic scene.node_tree API
    # compositor_build.py targets doesn't exist -- we must fail loudly
    # rather than silently build a broken graph. See module docstring and
    # README "Blender 5.0+ compositor". If you run this same test suite
    # against a real Blender < 5.0, this test should be skipped/updated
    # since ensure_compositor_tree() will succeed there instead.
    assert hasattr(clean_scene, "compositing_node_group"), (
        "this test's premise (Blender >= 5.0 API) no longer holds -- "
        "update it for the Blender version actually running it"
    )
    with pytest.raises(RuntimeError, match="compositing_node_group"):
        cb.ensure_compositor_tree(clean_scene)


def test_clear_managed_nodes_only_removes_prefixed_nodes():
    tree = bpy.data.node_groups.new("T", "CompositorNodeTree")
    ours = tree.nodes.new("CompositorNodeAlphaOver")
    ours.name = cb._node_name("Over", "0", "Foo")
    other = tree.nodes.new("CompositorNodeAlphaOver")
    other.name = "MyOwnNode"

    cb._clear_managed_nodes(tree)

    remaining = {n.name for n in tree.nodes}
    assert remaining == {"MyOwnNode"}
