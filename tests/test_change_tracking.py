from cycles_volume_override import change_tracking


def setup_function(_):
    change_tracking.reset()


def test_first_call_reports_changed_true():
    v = change_tracking.version_for("obj-1", "params", {"density_scale": 1.0})
    assert v.changed is True
    assert isinstance(v.hash, str) and v.hash


def test_identical_value_reports_changed_false():
    change_tracking.version_for("obj-1", "params", {"density_scale": 1.0})
    v2 = change_tracking.version_for("obj-1", "params", {"density_scale": 1.0})
    assert v2.changed is False


def test_different_value_reports_changed_true_and_different_hash():
    v1 = change_tracking.version_for("obj-1", "params", {"density_scale": 1.0})
    v2 = change_tracking.version_for("obj-1", "params", {"density_scale": 2.0})
    assert v2.changed is True
    assert v1.hash != v2.hash


def test_categories_are_independent_per_object():
    change_tracking.version_for("obj-1", "params", {"a": 1})
    dataset_v = change_tracking.version_for("obj-1", "dataset", {"path": "/x"})
    # dataset category has never been seen for obj-1 -- independent of params
    assert dataset_v.changed is True


def test_objects_are_independent():
    change_tracking.version_for("obj-1", "params", {"a": 1})
    v = change_tracking.version_for("obj-2", "params", {"a": 1})
    assert v.changed is True  # same value, different object -- no shared history


def test_hash_is_stable_regardless_of_dict_key_order():
    v1 = change_tracking.version_for("obj-1", "params", {"a": 1, "b": 2})
    change_tracking.reset()
    v2 = change_tracking.version_for("obj-1", "params", {"b": 2, "a": 1})
    assert v1.hash == v2.hash


def test_reset_clears_history():
    change_tracking.version_for("obj-1", "params", {"a": 1})
    change_tracking.reset()
    v = change_tracking.version_for("obj-1", "params", {"a": 1})
    assert v.changed is True
