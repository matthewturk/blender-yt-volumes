import bpy
import numpy as np
import pytest

from cycles_yt import external_render as er
from cycles_yt import operators, sample_renderers

RES_X, RES_Y = 33, 25  # odd dims -> a pixel sits exactly on the optical axis
CENTER_COL, CENTER_ROW = 16, 12


@pytest.fixture
def persp_camera_params():
    """A real PERSP CameraParams (real projection_matrix from bpy), camera
    at world (0, 0, 10), identity rotation -> looks straight down -Z.
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    cam_data = bpy.data.cameras.new("C")
    cam_data.type = "PERSP"
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("C", cam_data)
    cam_obj.location = (0.0, 0.0, 10.0)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    bpy.context.view_layer.update()
    return operators.build_camera_params(scene, cam_obj)


def _volume_params(density_scale=0.4, color=(1.0, 1.0, 1.0)):
    """A 2x2x2 box centered at world (0, 0, 5) -- so along the camera's
    straight-down-the-axis ray, it spans planar depth [4, 6] (camera at
    z=10): near face at world z=6 -> depth 4, far face at z=4 -> depth 6.
    """
    matrix_world = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 5], [0, 0, 0, 1]]
    bbox_local = [
        [x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)
    ]
    return er.VolumeObjectParams(
        matrix_world=matrix_world,
        bounding_box_local=bbox_local,
        custom_params={"density_scale": density_scale, "color": color},
        params_version=er.VersionInfo(hash="x", changed=True),
        dataset=er.DatasetParams(
            dataset_id="obj", dataset_kind="", dataset_path=None,
            is_blender_volume_sequence=False, sequence_frame_start=None,
            sequence_frame_duration=None, sequence_frame_offset=None,
            sequence_mode=None, resolved_sequence_frame=None, grid_names=[],
            version=er.VersionInfo(hash="y", changed=True),
        ),
    )


def _depth_buffer(fill_value):
    data = np.full((RES_Y, RES_X), fill_value, dtype=np.float32)
    return er.DepthBuffer(
        data=data, depth_convention="PLANAR", no_hit_value=1e10,
        row_order="BOTTOM_LEFT_ORIGIN", units="meters",
    )


def _request(camera, depth, volume):
    return er.RenderRequest(
        camera=camera, depth=depth, volume=volume,
        frame=er.FrameParams(frame_number=1, scene_time_seconds=0.0),
    )


def test_unoccluded_center_pixel_gets_expected_alpha(persp_camera_params):
    depth = _depth_buffer(1e10)  # no solid anywhere -- pure sky/no-hit
    volume = _volume_params(density_scale=0.4)
    result = sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))

    # path length through the box along the optical axis = 6 - 4 = 2;
    # alpha = clip(2 * 0.4, 0, 1) = 0.8
    alpha = result.rgba[CENTER_ROW, CENTER_COL, 3]
    assert alpha == pytest.approx(0.8, abs=1e-4)
    # premultiplied, color=(1,1,1) -> color channels == alpha
    assert result.rgba[CENTER_ROW, CENTER_COL, 0] == pytest.approx(0.8, abs=1e-4)
    assert result.alpha_mode == "PREMULTIPLIED"
    assert result.color_space == "LINEAR"


def test_corner_pixel_misses_the_box_entirely(persp_camera_params):
    depth = _depth_buffer(1e10)
    volume = _volume_params()
    result = sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))
    assert result.rgba[0, 0, 3] == 0.0


def test_solid_fully_in_front_occludes_volume_completely(persp_camera_params):
    # Solid at depth 3.0 -- closer to camera than the box's near face (4.0).
    depth = _depth_buffer(3.0)
    volume = _volume_params(density_scale=0.4)
    result = sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))
    assert result.rgba[CENTER_ROW, CENTER_COL, 3] == 0.0


def test_solid_partially_poking_into_volume_clips_path_length(persp_camera_params):
    # Solid at depth 5.0 -- inside the box's [4, 6] depth range.
    depth = _depth_buffer(5.0)
    volume = _volume_params(density_scale=0.4)
    result = sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))
    # clipped path length = 5 - 4 = 1; alpha = clip(1 * 0.4, 0, 1) = 0.4
    assert result.rgba[CENTER_ROW, CENTER_COL, 3] == pytest.approx(0.4, abs=1e-4)


def test_zero_density_scale_gives_zero_alpha(persp_camera_params):
    depth = _depth_buffer(1e10)
    volume = _volume_params(density_scale=0.0)
    result = sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))
    assert result.rgba[CENTER_ROW, CENTER_COL, 3] == 0.0


def test_non_planar_depth_convention_raises(persp_camera_params):
    depth = _depth_buffer(1e10)
    depth.depth_convention = "EUCLIDEAN"
    volume = _volume_params()
    with pytest.raises(NotImplementedError):
        sample_renderers.constant_density_renderer(_request(persp_camera_params, depth, volume))


def test_unsupported_panorama_subtype_raises():
    camera = er.CameraParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        projection="PANO", lens_mm=None, sensor_width_mm=36.0, sensor_height_mm=24.0,
        sensor_fit="AUTO", shift_x=0.0, shift_y=0.0, ortho_scale=None,
        clip_start=0.1, clip_end=1000.0, resolution_x=RES_X, resolution_y=RES_Y,
        pixel_aspect_x=1.0, pixel_aspect_y=1.0, projection_matrix=None,
        panorama=er.PanoramicLensParams(
            panorama_type="FISHEYE_EQUIDISTANT", fisheye_lens_mm=10.5, fisheye_fov_rad=3.14,
            fisheye_polynomial_k0=0.0, fisheye_polynomial_k1=0.0, fisheye_polynomial_k2=0.0,
            fisheye_polynomial_k3=0.0, fisheye_polynomial_k4=0.0,
            latitude_min_rad=-1.57, latitude_max_rad=1.57,
            longitude_min_rad=-3.14, longitude_max_rad=3.14,
            central_cylindrical_radius=0.0, central_cylindrical_range_u_min=0.0,
            central_cylindrical_range_u_max=0.0, central_cylindrical_range_v_min=0.0,
            central_cylindrical_range_v_max=0.0,
        ),
        dof=er.DepthOfFieldParams(
            use_dof=False, focus_distance_m=10.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    depth = _depth_buffer(1e10)
    volume = _volume_params()
    with pytest.raises(NotImplementedError):
        sample_renderers.constant_density_renderer(_request(camera, depth, volume))


def test_equirectangular_forward_direction_hits_centered_volume():
    """Pure-Python, no bpy: equirectangular doesn't need a projection_matrix,
    just matrix_world -- constructed by hand here, identity rotation.
    """
    camera = er.CameraParams(
        matrix_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]],
        projection="PANO", lens_mm=None, sensor_width_mm=36.0, sensor_height_mm=24.0,
        sensor_fit="AUTO", shift_x=0.0, shift_y=0.0, ortho_scale=None,
        clip_start=0.1, clip_end=1000.0, resolution_x=RES_X, resolution_y=RES_Y,
        pixel_aspect_x=1.0, pixel_aspect_y=1.0, projection_matrix=None,
        panorama=er.PanoramicLensParams(
            panorama_type="EQUIRECTANGULAR", fisheye_lens_mm=0.0, fisheye_fov_rad=0.0,
            fisheye_polynomial_k0=0.0, fisheye_polynomial_k1=0.0, fisheye_polynomial_k2=0.0,
            fisheye_polynomial_k3=0.0, fisheye_polynomial_k4=0.0,
            latitude_min_rad=-1.57, latitude_max_rad=1.57,
            longitude_min_rad=-3.14, longitude_max_rad=3.14,
            central_cylindrical_radius=0.0, central_cylindrical_range_u_min=0.0,
            central_cylindrical_range_u_max=0.0, central_cylindrical_range_v_min=0.0,
            central_cylindrical_range_v_max=0.0,
        ),
        dof=er.DepthOfFieldParams(
            use_dof=False, focus_distance_m=10.0, aperture_fstop=2.8,
            aperture_blades=0, aperture_ratio=1.0, aperture_rotation_rad=0.0,
        ),
    )
    depth = _depth_buffer(1e10)
    volume = _volume_params(density_scale=0.4)  # box centered at world (0,0,5) -- forward (-Z) from camera at z=10
    result = sample_renderers.constant_density_renderer(_request(camera, depth, volume))
    # forward (-Z, camera-local) -> u=0.5, v_from_top=0.5 -> center pixel
    assert result.rgba[CENTER_ROW, CENTER_COL, 3] == pytest.approx(0.8, abs=1e-3)
    # directly behind the camera (+Z, u~0/1) should miss the box entirely
    assert result.rgba[CENTER_ROW, 0, 3] == 0.0
