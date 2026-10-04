"""Deep Zoom: author camera paths through deeply nested AMR data in Blender.

See README.md. Everything float64 lives in `shot_space` and the scene's
`deep_zoom` settings; Blender only ever holds small offsets from the anchor.
"""

from . import authoring


def register():
    authoring.register()


def unregister():
    authoring.unregister()
