# Depth camera sim-to-real: paired real and simulated depth

Thirty paired captures for studying depth-sensor noise between simulation and the real world. Each capture has a real
RGB-D frame and an Isaac Sim render of the same tabletop scene from the same camera pose, so the simulated depth can be
used as approximate ground truth for the real depth.

| camera | shots | image (all aligned files) | distance to table | real depth source |
|---|---|---|---|---|
| Intel RealSense D435i | 10 | 640x360, fx 465.6 | about 0.75 m | librealsense 2.59.2, no post-processing |
| Stereolabs ZED Mini (robot head) | 10 | 672x376, fx 367.0 | about 0.71 m | ZED SDK 5.0.7, NEURAL mode |
| Intel RealSense D405 (robot wrist) | 10 | 424x240, fx 223.4 | about 0.29 m | realsense2_camera (ROS 2), no post-processing |

Scenes: a black table with two or three grocery items (snack boxes, cup noodles, cans). The robot is a ROBOTIS AI Worker.

## Layout

```
data/<camera>/<01..10>/     one capture per folder (camera = d435 | zed_mini | d405)
tools/load.py               read a capture as numpy arrays
tools/evaluate.py           pose alignment + error statistics (reproduces results/stats_*.json when run with --mono)
results/stats_<camera>.json   error statistics of every depth estimate vs ground truth (raw, CDM, sim extension, monocular models)
mono/<camera>/<model>/<01..10>.npy   the monocular predictions themselves (float32 metres, same size and K as the data)
```

## Files in a capture

| file | what it is | d435 | zed_mini | d405 |
|---|---|---|---|---|
| `real_rgb.png` | real colour image, reprojected to the simulated camera `K` | yes | yes | yes |
| `real_depth_mm.png` | real sensor depth, reprojected to `K` (nearest neighbour) | yes | yes | yes |
| `real_rgb_sensor.png` | real colour image as delivered by the sensor | no | yes | yes |
| `real_depth_sensor.png` | real depth as delivered by the sensor (own intrinsics, see `meta.json`) | yes | yes | yes |
| `sim_rgb.png` | Isaac Sim render | yes | yes | yes |
| `sim_depth_mm.png`, `sim_depth_m.npy` | simulation ground-truth depth (distance to image plane) | yes | yes | yes |
| `cdm_depth_mm.png`, `cdm_depth_m.npy` | [Camera Depth Model](https://huggingface.co/depth-anything) output from `real_rgb` + `real_depth_mm` | png only | yes | yes |
| `sim_sdk_depth_mm.png` | simulated stereo pair -> ZED Isaac Sim extension -> ZED SDK depth | no | yes | no |
| `sim_sdk_conf.png`, `sim_sdk_left.png` | ZED SDK confidence map and the left image the SDK received | no | yes | no |
| `meta.json` | intrinsics, sensor settings, simulation camera and scene info | yes | yes | yes |

Conventions

- Depth PNGs are 16-bit, in millimetres, `0` = invalid. `.npy` files are float32 metres.
- `real_rgb`, `real_depth_mm`, `sim_*`, `cdm_*` and `sim_sdk_*` share one pinhole camera, `K` in `meta.json`.
- PNG colour images are stored the usual way; `tools/load.py` returns RGB.

## Quick start

```bash
pip install numpy opencv-python scipy
python tools/load.py                # one line per capture
python tools/evaluate.py d405       # error statistics for one camera
```

```python
from tools.load import load_shot
s = load_shot("data/zed_mini/03")
err_mm = (s["raw_m"] - s["gt_m"])[(s["raw_m"] > 0) & (s["gt_m"] > 0)] * 1000
```

## Real and simulated frames are close, not identical

The real camera and the scene were placed by hand to match the simulation. A residual pose remains, the same for all
ten shots of a camera. `tools/evaluate.py` estimates it by fitting the real raw-depth point cloud to the ground-truth
surface (point-to-plane ICP):

| camera | rotation | translation (mm) |
|---|---|---|
| d435 | 3.21 deg | 39.0, -30.7, -10.4 |
| zed_mini | 2.50 deg | -34.4, -7.0, 3.5 |
| d405 | 0.95 deg | -1.0, 3.2, -3.5 |

Object placement was also done by hand. Only the table and the objects correspond between real and sim; the room
behind them does not (the simulation has an empty floor).

## Reference results

Median absolute error `|e|` and 90th percentile against ground truth, in millimetres, over ten shots, after applying
the pose above. "Tail points" are points near an object that lie more than 20 mm from any ground-truth surface.
"Lean" is the error of the tilt of standing objects.

### D435
| depth | holes % | table \|e\| | table p90 | object \|e\| | object p90 | lean max ° | tail points % |
|---|---|---|---|---|---|---|---|
| raw | 2.2 | 4.7 | 13.4 | 9.3 | 19.7 | 16.1 | 8.3 |
| CDM | 0.0 | 5.7 | 31.7 | 13.4 | 36.9 | 42.6 | 21.7 |
| best monocular + scale (UniDepthV2) | 0.0 | 4.2 | 10.0 | 12.1 | 29.1 | 32.7 | 9.4 |

### ZED Mini
| depth | holes % | table \|e\| | table p90 | object \|e\| | object p90 | lean max ° | tail points % |
|---|---|---|---|---|---|---|---|
| raw | 0.4 | 2.9 | 11.0 | 7.0 | 14.8 | 14.9 | 6.8 |
| CDM (ZED 2i model) | 0.0 | 3.6 | 9.1 | 6.8 | 26.5 | 48.0 | 11.7 |
| sim + ZED extension | 0.1 | 6.2 | 18.0 | 1.9 | 6.4 | 6.1 | 3.4 |
| best monocular + scale (UniDepthV2) | 0.0 | 10.2 | 20.5 | 6.8 | 27.0 | 29.5 | 13.1 |

### D405
| depth | holes % | table \|e\| | table p90 | object \|e\| | object p90 | lean max ° | tail points % |
|---|---|---|---|---|---|---|---|
| raw | 10.4 | 1.9 | 7.0 | 3.8 | 8.6 | 10.5 | 1.9 |
| CDM | 0.0 | 1.8 | 15.5 | 5.8 | 14.9 | 32.9 | 4.9 |
| best monocular + scale (MoGe2) | 0.0 | 6.5 | 50.7 | 10.3 | 35.0 | 81.5 | 29.2 |

Monocular models (MoGe-2, UniDepth V2, Depth Anything 3 Metric / Nested, Depth Anything V2, Depth Pro, Metric3D v2)
were given one real RGB image; "+ scale" multiplies the prediction by `median(raw / prediction)`, so those rows are
not sensor-free results. All seven are in `results/stats_<camera>.json`, and the predictions are in `mono/`
(`_K` in a model name means the known focal length was given to the model). To recompute their statistics:

```bash
python tools/evaluate.py zed_mini --mono mono/zed_mini
```

## Known limitations

- **Pose from raw depth.** The alignment is estimated from raw depth, so a constant bias of raw depth is absorbed by
  it. Table-surface numbers favour raw depth slightly; object-surface and tail-point numbers are less affected.
- **ZED sim-extension depth is rescaled.** The ZED SDK on the simulation side reported fx 264.9 px while the render used
  fx 367.0 px, and the capture server multiplied the SDK depth by 1.385 (`meta.json` -> `sim.sdk`). Whether this is the
  intended correction was not verified.
- **Different ZED SDK versions.** Real: 5.0.7. Simulation: 5.6.0. Both NEURAL mode, confidence 95.
- **CDM on ZED Mini uses the ZED 2i model.** There is no official ZED Mini checkpoint. It was run offline after capture.
- **D405 sees its gripper.** The fingers appear at the left edge of every D405 image; `tools/evaluate.py` masks them.
  Raw D405 depth also has a band of invalid pixels at the left edge and a few pixels that jump to several metres.
- **D435 set has no `real_rgb_sensor.png`** and its CDM output is PNG only.
- **Reprojection.** `real_rgb` is resampled bilinearly and `real_depth_mm` by nearest neighbour; use the `*_sensor`
  files with their own intrinsics if you need untouched sensor data.
- **Small and simple.** Ten shots per camera, two or three objects, one table. ZED Mini shot 01 has a person at the
  left edge of the real image.

## License

Code (`tools/`): MIT, see `LICENSE`. Data (`data/`, `mono/`, `results/`): CC BY 4.0, see `LICENSE-DATA`.

The images show commercial product packaging; trademarks belong to their owners. `cdm_*` files and `mono/` are outputs
of third-party models (Camera Depth Models, MoGe-2, UniDepth V2, Depth Anything 3 / V2, Depth Pro, Metric3D v2); check
the licences of those models before using their outputs commercially.
