# Trajectory-constrained exploration

One Gaussian scene, and the path the cameras that trained it travelled. Load a
`cameras.json` and walk the scene with the mouse and <kbd>W</kbd><kbd>A</kbd>
<kbd>S</kbd><kbd>D</kbd> — held inside a tube around that path, because a
Gaussian reconstruction is only trustworthy where the training cameras looked.

**This milestone is single-scene.** A scene gallery, multi-scene routing and
comparing reconstructions are the *next* milestone and are not implemented here:
the viewer shows one scene at a time and one path belonging to it.

**It is not a tour.** There is no Play button, no speed multiplier and no
loop. The reason is in the data: see [Places, not poses](#places-not-poses).

## Supported input

The original INRIA/Graphdeco `cameras.json`, written by
`utils/camera_utils.py::camera_to_JSON` in
[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
and by every trainer that follows it. It is a JSON array:

```json
[
  {
    "id": 0,
    "img_name": "station_000_t000_0s_back.png",
    "width": 2048,
    "height": 2048,
    "position": [7.2384, -0.5022, 1.6313],
    "rotation": [
      [-0.0683, -0.0922,  0.9934],
      [ 0.4505,  0.8856,  0.1132],
      [-0.8901,  0.4553, -0.0190]
    ],
    "fy": 1024.0,
    "fx": 1024.0
  }
]
```

Every field above is required. Validation happens before anything is displayed,
and **the whole file is rejected or nothing is**: a single bad camera never
produces a half-loaded path.

| Rejected | Reason reported |
| --- | --- |
| not a JSON array, or empty | "a JSON array of camera objects" / "contains no cameras" |
| missing or non-finite `id`, `position`, `rotation`, `fx`, `fy` | names the camera index and the field |
| `width`, `height`, `fx`, `fy` not positive | names the camera index and the field |
| `rotation` not 3×3, or not a right-handed rotation matrix | "not a right-handed rotation matrix (determinant …)" |
| larger than 64 MiB | "too large to be a cameras.json" |

Duplicate `id` values are **not** fatal — some exports repeat them. The cameras
are kept, keyed internally by their position in the source array, and a notice
says so.

## Coordinate convention

This is the part that is easy to get wrong by guessing, so it was read out of the
official implementation rather than inferred from the field names.

1. `camera_to_JSON` builds `Rt` from `camera.R.transpose()` and `camera.T`, then
   writes `np.linalg.inv(Rt)`. Its local variable is called `W2C`, which is
   misleading: `utils/graphics_utils.py::getWorld2View2` performs the *identical*
   construction and names that same inverse **`C2W`**.
2. `camera.R` is the camera-to-world rotation and `camera.T` the world-to-camera
   translation — `scene/dataset_readers.py` builds them as
   `R = np.transpose(qvec2rotmat(extr.qvec))` and `T = np.array(extr.tvec)`, with
   the comment *"R is stored transposed due to 'glm' in CUDA code"*.

So, in `cameras.json`:

* **`position` is the camera centre in world space** — the same world the splats
  are in, with no extra transform.
* **`rotation` is the camera-to-world rotation, with the camera's axes as its
  columns**, in COLMAP/OpenCV orientation:

  | column | camera axis |
  | --- | --- |
  | 0 | **right** (+x) |
  | 1 | **down** (+y) |
  | 2 | **forward** (+z) |

Three.js uses the OpenGL convention — a camera looks down its own **−z** with
**+y** up — so the conversion is to negate columns 1 and 2:

```
three_right   =  column 0
three_up      = -column 1
three_forward =  column 2      (the camera looks along -z, so its -z axis is column 2)
```

That is exactly the inverse of Graphdeco's own OpenGL→COLMAP line, which the
official reader applies when it ingests NeRF transforms:

```python
# change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
c2w[:3, 1:3] *= -1
```

Vertical field of view comes from `fy` and the image **height**, following
`utils/graphics_utils.py::focal2fov`:

```
fovY = 2 * atan(height / (2 * fy))
```

`fx`, `width` and the field of view are used only to draw the debug frustums at
each raw pose. **The viewer keeps its own field of view throughout**: walking is
first-person exploration, not a reproduction of a training image, so nothing
here ever imposes a training camera's FOV on the canvas or letterboxes the
viewport to a training image's shape.

No sign in any of this was chosen by looking at a render.
`frontend/tests/camera-trajectory.spec.ts` pins position, forward, up and right
numerically against an asymmetric known pose whose transpose, inverse and
sign-flips all give different answers.

## Loading a path

**Locally.** Open a Gaussian scene, then **Open camera path** and pick a
`.json` file. The bytes stay in the browser: they are read, parsed and dropped.
Nothing is uploaded, nothing is written to SQLite or `localStorage`, and the
file handle does not survive a refresh — reopen the file after one.

A malformed file leaves the current scene *and* the currently loaded path
untouched, and reports why. Opening a different Gaussian scene clears the path,
because a path from one reconstruction means nothing in another.

**From a hosted scene.** A scene in `frontend/public/data/scenes.json` may add an
optional `trajectoryUrl`, and the viewer loads it with the scene:

```json
{
  "id": "campus-walk",
  "name": "Campus walk",
  "url": "https://static.example.edu/scenes/campus-lod.rad",
  "trajectoryUrl": "https://static.example.edu/scenes/cameras.json"
}
```

Only `http`/`https` URLs ending in `.json` are accepted, resolved against the
application base like the RAD URL. A `trajectoryUrl` that fails to load, or is
rejected, drops only the path — the scene still opens. Cross-origin hosting
needs the usual `Access-Control-Allow-Origin`; see
[`hosted-scenes.md`](hosted-scenes.md).

## Places, not poses

The real capture this was built against has **948 cameras** — and **158 places**.
Every place is photographed as a six-face cube: `_front`, `_back`, `_left`,
`_right`, `_up`, `_down`, all six sharing one centre.

```
station_000_t000_0s_back.png  |
station_000_t000_0s_down.png  |  one place,
station_000_t000_0s_front.png |  six poses,
station_000_t000_0s_left.png  |  one camera centre
station_000_t000_0s_right.png |
station_000_t000_0s_up.png    |
station_001_t000_5s_back.png     the rig moves 0.30 further on
```

Replaying those 948 poses in file order spins on the spot through six
orientations, steps 0.30 forward, and spins again — 158 times. That is not a
trajectory, and it is why the earlier playback control is gone.

So the cameras are collapsed to the distinct places they were taken from, and
**those** are the trajectory. The panel says both numbers,
`948 cameras · 158 trajectory anchors`, because both are true and only one of
them is a path.

### How the places are found

By proximity, not by equality. The six positions at one place are **not**
bit-identical in the real export: the largest spread inside a station is
`2.3e-14`, float noise from the reconstruction, and all 948 position tuples are
distinct. Grouping by equality would find 948 places.

The tolerance is derived from the capture, never hard-coded in world units — a
scene in metres and the same scene in millimetres must collapse the same way:

```
tolerance = max( extent × 1/1000 ,          extent = diagonal of the camera-centre box
                 |largest coordinate| × 1e-9 ,
                 1e-9 )
```

On the real data that is `16.770 × 0.001 = 0.0168`, which sits comfortably
between the two things it has to separate:

| quantity | value | ratio to the tolerance |
| --- | --- | --- |
| largest spread inside one place | 2.3e-14 | ~10¹² below |
| smallest gap between two places | 0.1287 | 7.7× above |
| median gap between places | 0.2979 | 17.8× above |

The second and third terms are the floor. A capture taken from a *single* place
has an extent of only its own float noise, and a tolerance derived from that
would split one place into six; two coordinates that agree to one part in a
billion are the same point at any scene scale.

Matching is done on a uniform grid of `tolerance`-sized cells, so each camera
looks in the 27 cells around it and the cost is linear rather than quadratic.

Places keep **file order**. Station numbers in `img_name` are parsed, but only
to *check* that order: if they run backwards the panel says so and the file
order is still used, because no schema guarantees the names exist at all.

## The tube

The region the viewer may enter is defined once, as a distance:

```
allowed = { p : distance(p, polyline through the anchors) ≤ radius }
```

Anything outside it is projected back onto the boundary. Two behaviours fall out
of that definition rather than being coded as special cases:

* **Sliding.** Only the part of the motion that left the tube is removed, so
  walking into the wall at an angle keeps making ground along the path instead
  of stopping dead.
* **Rounded ends.** The closest point on a segment is clamped to its endpoints,
  so past the last anchor the allowed set is already a ball of the same radius.

**Radius and speed come from the capture's own spacing**, so a scene in
millimetres is as walkable as one in metres:

| | formula | on the real data |
| --- | --- | --- |
| default radius | `median gap × 0.5` | 0.1490 |
| radius range | `extent/400` … `extent/10` | 0.0419 … 1.6770 |
| default speed | `median gap × 2` per second | 0.5958 /s |
| speed range | `× 0.25` … `× 12` | 0.0745 … 3.575 /s |

Both are adjustable in the panel while walking, and shrinking the radius pulls a
standing camera back in rather than waiting for the next keypress.

### Crossings

A capture path comes back on itself. A plain nearest-point search would hand the
walker whichever branch is closest at that instant, which teleports them across
to the other leg every time the two pass near each other. So the search is
windowed: only the **8 segments either side of the one the walker is on** are
considered, and that segment is carried from frame to frame. Eight segments is
about 2.4 world units here — far more than one frame of walking, so the window
cannot be outrun. A global search is used only when *entering* the mode and for
**Return to nearest anchor**, where there is no previous segment to honour.

## Walking

**Explore trajectory** hands the camera to the walker and switches OrbitControls
off — only while it is on.

| | |
| --- | --- |
| look | mouse (click the scene to capture it), or <kbd>←</kbd><kbd>→</kbd> and <kbd>Page Up</kbd>/<kbd>Page Down</kbd> |
| move | <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>, or <kbd>↑</kbd><kbd>↓</kbd> |
| down / up | <kbd>Q</kbd> / <kbd>E</kbd> |
| faster | <kbd>Shift</kbd> (3×) |
| release the mouse | <kbd>Esc</kbd> |
| leave the mode | <kbd>Esc</kbd> again, or **Stop exploring** |

Keys are accepted whenever the pointer is locked **or** the canvas has focus.
Pointer lock needs a user gesture and is refused outright in some environments;
a walk that only works with a captured mouse would not be usable from a keyboard
at all.

**Entering keeps your view.** The camera does not turn, and it lands somewhere
it can move *from*: the nearest point of the path itself if you were outside the
tube, or exactly where you already were if you were inside it. The literal
nearest valid point from an overview camera is a point on the *boundary*, which
would start every walk pinned against the wall. **Choosing an anchor** — Previous,
Next, the slider, or clicking a marker — likewise moves where you stand and
never where you look: the poses at an anchor are cube faces, and adopting one of
them is the spinning this mode exists to replace.

Roll never accumulates. Yaw and pitch are stored as angles against the scene's
own up axis — the one the orientation controls set — and the orientation is
rebuilt from them every frame, so the horizon cannot drift. Pitch stops about 2°
short of straight up and straight down.

The panel shows, live: whether the boundary is holding you
(`Trajectory boundary`), how far from the path you are, and which anchor you are
nearest. **Return to nearest anchor** puts you back on the path, and is the way
out of a corner you have pressed yourself into.

## What it is drawn with

The **collapsed** path: one polyline through the 158 anchors, one `Points` for
the markers, one short `LineSegments` highlighting the segment you are on, and a
cross plus a larger dot on the selected anchor — so selection is never signalled
by colour alone. Five objects, whatever the camera count.

The 948 raw frustums are a **debug aid**, off by default, and the geometry is not
even built until the toggle is switched on. Drawing them by default stacks six
frustums on every marker and says nothing about where you may go. When they are
on they are one batched `LineSegments`: a `THREE.CameraHelper` or a DOM label per
camera would mean thousands of objects and thousands of draw calls paid by Spark
on every frame.

Nothing in the overlay applies a transform of its own: it is added to the same
scene as the splats, so the path, the markers and the Gaussians share exactly
one world space. The orientation controls (upright, front, roll, presets) move
only the *camera*, never the scene, so they move all three together by
construction.

## Known limitations

* **Only the Graphdeco `cameras.json` schema is read.** COLMAP `images.txt`,
  NeRF `transforms.json` and Spark's own formats are not. The canonical model
  carries an unused `timestampSeconds` so a format that has one can be added
  without changing the visualization or the walk.
* **The tube is around the path, not around the reconstruction.** It says
  nothing about what the cameras could *see*: a place the rig stood next to a
  wall gets the same sphere of freedom as one in the open. A visibility- or
  depth-aware region would be a better boundary and is not implemented.
* **The radius is uniform.** It does not widen where the capture is dense or
  narrow where it is sparse.
* **Collapsing is positional only.** Two genuinely different places closer
  together than the tolerance would merge, and the panel would report fewer
  anchors than the capture has. On this data the margin is 7.7×.
* **The path is attached to Gaussian scenes only.** The point-cloud viewer is
  unchanged and has no path control.
* **No image thumbnails.** `img_name` is shown as text; the training images
  themselves are not loaded or displayed.
* **Anchor labels are not drawn in the scene.** With hundreds of places they are
  unreadable clutter; the panel reports the index and the station instead.
* **A path is not persisted.** Reopen the file after a refresh.
* **Nothing validates the path against the scene.** Loading a `cameras.json`
  from a different reconstruction produces a path in the wrong place rather than
  an error; the viewer has no way to tell.
* **Spark logs one unhandled rejection when a Gaussian scene is replaced while
  the camera has just been moving.** It comes from a sort or level-of-detail
  worker call already in flight when `SparkRenderer.dispose()` terminates the
  worker. The viewer quiesces the renderer first, which removes one of the two
  that used to appear; the remaining one is inside Spark and cannot be caught
  from here. It is console-only — the scene, the canvas and the overlay are all
  released correctly.
* **Multi-scene exploration is the next milestone**, not part of this one.
