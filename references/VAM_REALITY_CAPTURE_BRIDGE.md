# VaM Reality Capture Bridge

Prototype source:

`vam_runtime/bepinex/VaMRealityCaptureBridge/VaMRealityCaptureBridge.cs`

The bridge is manual-first:

- User loads the scene/animation manually.
- Bridge captures the current viewport.
- It does not load scenes.
- It does not move atoms/controllers.
- It does not control Timeline.
- It does not save the VaM scene.

Endpoints:

- `GET /status`
- `POST /capture_frames`

The Python client handles unavailable bridge status by writing a blocked report
instead of crashing.
