import numpy as np

from yt_volumes import external_render as er

IDENTITY_4X4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _default_dof():
    return er.DepthOfFieldParams(
        use_dof=False,
        focus_distance_m=10.0,
        aperture_fstop=2.8,
        aperture_blades=0,
        aperture_ratio=1.0,
        aperture_rotation_rad=0.0,
    )


def _persp_camera(resolution_x=1920, resolution_y=1080):
    return er.CameraParams(
        matrix_world=IDENTITY_4X4,
        projection="PERSP",
        lens_mm=50.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_fit="AUTO",
        shift_x=0.0,
        shift_y=0.0,
        ortho_scale=None,
        clip_start=0.1,
        clip_end=1000.0,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
        projection_matrix=IDENTITY_4X4,
        panorama=None,
        dof=_default_dof(),
    )


def test_dataclass_fields_match_contract():
    cam = _persp_camera()
    assert cam.projection == "PERSP"
    assert cam.ortho_scale is None
    assert cam.panorama is None
    assert cam.projection_matrix == IDENTITY_4X4


def test_panoramic_lens_params_is_none_unless_pano():
    persp = _persp_camera()
    assert persp.panorama is None

    pano = er.CameraParams(
        matrix_world=IDENTITY_4X4,
        projection="PANO",
        lens_mm=None,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        sensor_fit="AUTO",
        shift_x=0.0,
        shift_y=0.0,
        ortho_scale=None,
        clip_start=0.1,
        clip_end=1000.0,
        resolution_x=128,
        resolution_y=64,
        pixel_aspect_x=1.0,
        pixel_aspect_y=1.0,
        projection_matrix=None,  # PANO has no single-matrix representation
        panorama=er.PanoramicLensParams(
            panorama_type="EQUIRECTANGULAR",
            fisheye_lens_mm=10.5,
            fisheye_fov_rad=3.14159,
            fisheye_polynomial_k0=0.0, fisheye_polynomial_k1=0.0,
            fisheye_polynomial_k2=0.0, fisheye_polynomial_k3=0.0,
            fisheye_polynomial_k4=0.0,
            latitude_min_rad=-1.5708, latitude_max_rad=1.5708,
            longitude_min_rad=-3.14159, longitude_max_rad=3.14159,
            central_cylindrical_radius=0.0,
            central_cylindrical_range_u_min=0.0, central_cylindrical_range_u_max=0.0,
            central_cylindrical_range_v_min=0.0, central_cylindrical_range_v_max=0.0,
        ),
        dof=_default_dof(),
    )
    assert pano.projection_matrix is None
    assert pano.panorama.panorama_type == "EQUIRECTANGULAR"


def _make_minimal_request(resolution=(2, 2)):
    h, w = resolution
    depth = er.DepthBuffer(
        data=np.full((h, w), 10.0, dtype=np.float32),
        depth_convention="PLANAR",
        no_hit_value=1e10,
        row_order="BOTTOM_LEFT_ORIGIN",
        units="meters",
    )
    cam = _persp_camera(resolution_x=w, resolution_y=h)
    volume = er.VolumeObjectParams(
        matrix_world=IDENTITY_4X4,
        bounding_box_local=[[0, 0, 0]] * 8,
        custom_params={"color": (0.4, 0.6, 1.0), "density_scale": 1.0},
        params_version=er.VersionInfo(hash="deadbeef", changed=True),
        dataset=er.DatasetParams(
            dataset_id="test-obj-uuid",
            dataset_kind="",
            dataset_path=None,
            is_blender_volume_sequence=False,
            sequence_frame_start=None,
            sequence_frame_duration=None,
            sequence_frame_offset=None,
            sequence_mode=None,
            resolved_sequence_frame=None,
            grid_names=[],
            version=er.VersionInfo(hash="cafef00d", changed=True),
        ),
    )
    frame = er.FrameParams(frame_number=1, scene_time_seconds=0.0)
    return er.RenderRequest(camera=cam, depth=depth, volume=volume, frame=frame)
