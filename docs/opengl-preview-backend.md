# AtomStudio OpenGL Preview Backend 设计文档

## Status
- Draft
- Date: 2026-05-10
- Owner: AtomStudio core

## Executive Summary
AtomStudio 的实时预览需要从当前的 VisPy `visuals.InstancedMesh` 路线切换到自研 OpenGL renderer。

原因不是“VisPy 不使用 OpenGL”，而是当前路径依赖的是通用 scene graph 和通用 visual。它能快速做 demo，但对晶体/分子结构可视化的关键需求控制不足：

- GL state 由通用 visual 隐式管理，容易出现 blending、depth、shading 行为不可控。
- Atom、bond、cell、selection 的渲染逻辑被塞进 `app/preview_canvas.py`，边界混乱。
- 旋转、缩放、样式更新和 picking 混在 UI widget 中，性能问题难定位。
- 继续补 VisPy visual 会形成新的过渡层，后续仍然要推倒重来。

本设计直接确定长期路线：

`PySide6 + QOpenGLWidget + PyOpenGL + NumPy + GLSL`

第一版仍用 Python 写 OpenGL 调用，但不再使用 VisPy 作为预览主线。后续只有在明确测到 Python 调用层成为瓶颈时，才把稳定后的 renderer 平移到 C++/Rust。

## Goals
- 建立长期可维护的专用 OpenGL preview backend。
- `app` 只负责 Qt 界面和事件转发，不负责 mesh、shader、GL state。
- `preview` 只消费统一 `RenderScene` / preview scene 数据，不重新解析 `Structure + Config`。
- Atom、bond、cell 按对象类型批量绘制，不允许每个对象一个 visual/widget/object。
- 相机拖动只更新矩阵，不重建 mesh、不重新上传不变 buffer。
- 首版目标是 VESTA 风格的清晰、稳定、快速预览，而不是 Blender 级材质。

## Non-Goals
- 不在第一版实现体数据、isosurface、透明排序。
- 不在第一版追求 Blender Cycles 材质一致性。
- 不在第一版实现 GPU picking；先用 CPU ray picking。
- 不保留 VisPy fallback 作为长期主线。
- 不继续向 `app/preview_canvas.py` 增加渲染细节。

## Technology Choice

### Chosen
- UI: `PySide6`
- Preview widget: `QOpenGLWidget`
- OpenGL binding: `PyOpenGL`
- Data arrays: `NumPy`
- Shader: GLSL source files in repo

### Why Not VisPy Mainline
VisPy 可以创建 OpenGL context，也可以写 `gloo.Program`，但继续依赖它会保留一层额外抽象：

- backend 差异仍然存在。
- GL state 仍可能被框架层影响。
- 与 Qt 事件、鼠标、FBO、MSAA 的边界不如 `QOpenGLWidget` 明确。
- 后续迁移 C++/Rust 时仍要重新拆除 VisPy 依赖。

因此不再把 VisPy 作为最终 preview backend。

### Why Not Immediate C++
当前项目主体是 Python。立刻上 C++ 会增加构建、打包、Qt ABI、Windows/Linux OpenGL loader、pybind11 边界等复杂度。

第一阶段先用 Python 直接管理 OpenGL 资源。只要做到 buffer 稳定、draw call 批量化、交互时只更新矩阵，Python 调用层足够支撑首版结构预览。

## Target Module Layout

```text
src/atomstudio/app/
  preview_widget.py
    QOpenGLWidget 壳，负责 Qt 生命周期、鼠标键盘事件、resize、菜单/toolbar 调用。

src/atomstudio/preview/
  controller.py
    Preview scene 状态、selection 状态、相机状态协调。

  camera.py
    Orbit/pan/zoom/fit/projection math。

  picking.py
    CPU ray picking for atoms/bonds。

  selection.py
    选择集合、active selection、selection payload。

  gl/
    renderer.py
      OpenGLRenderer 总入口。

    resources.py
      ShaderProgram、Buffer、VertexArray、GL resource 生命周期。

    batches.py
      AtomBatch、BondBatch、CellBatch、SelectionBatch。

    mesh.py
      sphere/cylinder/cell helper mesh generation。

    state.py
      GL state setup: depth, culling, MSAA, blending policy。

    shaders/
      atom.vert
      atom.frag
      bond.vert
      bond.frag
      cell.vert
      cell.frag
      selection.vert
      selection.frag
```

## Ownership Boundaries

### `app.preview_widget`
Allowed:
- Create/destroy `QOpenGLWidget`.
- Forward `initializeGL`, `resizeGL`, `paintGL`.
- Translate Qt mouse/keyboard events into controller commands.
- Show/hide overlay widgets such as axis marker.

Forbidden:
- Build sphere/cylinder mesh.
- Own shader source or GL buffers.
- Resolve materials, radius, representation.
- Do picking math directly.

### `preview.controller`
Allowed:
- Hold current preview scene.
- Hold current camera.
- Hold current selection.
- Tell backend when geometry/style changed.
- Tell backend when only camera changed.

Forbidden:
- Call raw OpenGL.
- Import Qt widget classes.
- Resolve domain semantics from `Structure`.

### `preview.gl.renderer`
Allowed:
- Own OpenGL lifecycle.
- Own shader programs and draw passes.
- Receive prepared scene records/buffers.
- Draw atom/bond/cell/selection batches.

Forbidden:
- Know about file loading.
- Know about Blender render config export.
- Mutate `Structure`.
- Reach into app panels.

### `scene` / `preview data`
Allowed:
- Produce atom positions, radii, colors, material payloads, bond endpoints, bond colors.
- Represent semantic truth shared by preview and Blender.

Forbidden:
- Create OpenGL objects.
- Create Qt objects.
- Create `bpy` objects.

## Rendering Pipeline

```text
Structure + RenderJobConfig
        |
        v
SceneBuilder / preview data adapter
        |
        v
PreviewController.set_scene(...)
        |
        v
OpenGLRenderer.update_scene(...)
        |
        +--> AtomBatch uploads instance buffer
        +--> BondBatch uploads instance buffer
        +--> CellBatch uploads line/cylinder buffer
        +--> SelectionBatch uploads selected object buffer
        |
        v
QOpenGLWidget.paintGL()
        |
        v
OpenGLRenderer.draw(camera)
```

Camera-only interaction:

```text
mouse drag/wheel
    -> PreviewController updates CameraState
    -> QOpenGLWidget.update()
    -> paintGL()
    -> renderer.draw(new view/projection matrices)
```

No geometry buffer rebuild should happen during camera-only interaction.

## Draw Passes

### Pass 1: Opaque Atoms
- One sphere base mesh.
- Per-instance attributes:
  - position
  - radius
  - color
  - selection flag
  - material id or material compact parameters
- Shader:
  - model transform from instance attributes
  - simple Blinn-Phong or matcap lighting
  - depth write enabled
  - blending disabled

### Pass 2: Opaque Bonds
- One cylinder/tube base mesh or procedural tube.
- Per-instance attributes:
  - start
  - end
  - radius
  - color_a
  - color_b
  - style flags
  - selection flag
- Bicolor should be handled in shader by axial interpolation, not by overlapping capped half-cylinders.
- Depth write enabled.
- Blending disabled.

### Pass 3: Cell / Axes / Helper Lines
- Either GL lines with stable width limits or thin cylinders.
- Prefer thin cylinders if OpenGL core profile line width is inconsistent.

### Pass 4: Selection Highlight
- Depth-aware, not x-ray by default.
- First version:
  - selected atom shell: slightly expanded sphere, opaque or low-alpha but depth-tested.
  - selected bond: color/radius change only, no screen-space ring.
- Later:
  - outline pass using depth/normal edge detection.

### Pass 5: UI Overlay
- Axis overlay can stay Qt painter-based or become a tiny second OpenGL pass.
- It must not share world translation/scale, only camera orientation.

## GL State Policy

Default for structural objects:

```text
depth_test = true
depth_write = true
blend = false
cull_face = optional, default false until winding is verified
MSAA = enabled if available
clear_depth every frame
```

Transparent objects are not supported in v1. Do not add partial transparency to atom/bond preview until the opaque pipeline is stable.

## Camera and Projection

Camera responsibilities:
- orbit
- pan
- zoom
- fit to structure
- top/front/side/a/b/c views
- orthographic/perspective modes

Important rules:
- Zoom changes projection or camera distance, never object coordinates.
- Pan changes camera center, never object coordinates.
- Rotate changes view matrix, never object coordinates.
- Near/far planes must be recomputed from scene bounds to preserve depth precision.

Depth precision rule:

```text
near = max(scene_radius * 0.001, 0.001)
far = max(scene_radius * 8.0, near + 1.0)
```

This should be refined during implementation, but near must never be unnecessarily close to zero.

## Geometry Rules

### Atom
- Use shared UV sphere or icosphere mesh.
- LOD levels:
  - low: 8-12 segments during interaction
  - normal: 16-24 segments after interaction
  - high: optional still preview/screenshot

### Bond
- Do not generate separate visual/object per bond.
- Do not split bicolor into two capped cylinders.
- Use one bond instance and interpolate color in shader.
- For double/triple bonds:
  - generate multiple instances with stable offset basis.
  - offset basis must be view-independent or chemically consistent.
- For dashed/dotted styles:
  - first version can expand into multiple instances.
  - later version should use shader-space patterning.

### Cell
- First version can be cylinder edges.
- Avoid GL wide lines as the only implementation because driver support is inconsistent.

## Picking

First version: CPU ray picking.

Atom picking:
- Convert screen point to world ray.
- Intersect ray with atom spheres.
- Choose nearest positive hit depth.

Bond picking:
- Intersect ray against finite cylinder approximation or compute closest ray-segment distance.
- Use bond radius plus screen tolerance.
- Choose nearest hit depth.

Selection priority:
1. Atom hit
2. Bond hit

This matches user expectation for dense structures.

Later version: GPU color picking pass, only if CPU picking becomes too slow.

## Performance Rules

Hard rules:
- No mesh rebuild on camera movement.
- No Python loop per object during draw.
- No per-object OpenGL draw calls for atoms/bonds.
- No Qt repaint/update storm during mouse drag.
- Geometry/style buffer upload only when scene/style changes.
- Camera updates only update uniforms.

Target draw call count for basic scene:
- atoms: 1 draw call
- bonds: 1 draw call
- cell: 1 draw call
- selection: 1 draw call

Target performance:
- 200 atoms: smooth interaction on WSL/GWSL and native Linux.
- 2,000 atoms: usable interaction with normal LOD.
- 10,000 atoms: usable with lower LOD and optional bond culling.

## Quality Rules

To approach VESTA-style clarity:
- Prefer stable simple lighting over complex PBR.
- Use screen-space fixed light by default.
- Enable MSAA at widget/context level.
- Avoid blending for core structural geometry.
- Keep atom/bond colors saturated but not overexposed.
- Use depth-cueing later as a controlled pass, not as ad-hoc alpha.
- Selection highlight must be visible but not x-ray unless explicitly enabled.

## Implementation Phases

### Phase 0: Freeze Current VisPy Scope
Goal:
- Stop adding new rendering features to `app/preview_canvas.py`.

Tasks:
- Mark VisPy preview path as legacy.
- Add documentation comment in `preview_canvas.py` pointing to this design.
- Only bug fixes allowed in VisPy path.

Acceptance:
- Existing app still launches.
- No behavior change required.

### Phase 1: QOpenGLWidget Shell
Goal:
- Create a clean Qt/OpenGL widget with no molecule rendering yet.

Files:
- `src/atomstudio/app/preview_widget.py`
- `src/atomstudio/preview/gl/renderer.py`
- `src/atomstudio/preview/gl/state.py`

Tasks:
- Initialize OpenGL context.
- Enable depth test and MSAA.
- Clear background.
- Handle resize.
- Draw a test triangle or cube.

Acceptance:
- `atomstudio app --preview-backend opengl` opens a widget.
- No VisPy imports in new backend.
- Offscreen smoke test can instantiate widget where platform supports it.

### Phase 2: Camera System
Goal:
- Implement stable orbit/pan/zoom/fit independent of object geometry.

Files:
- `src/atomstudio/preview/camera.py`
- `src/atomstudio/app/preview_widget.py`

Tasks:
- View matrix.
- Projection matrix.
- Orthographic/perspective.
- Mouse rotate/pan/zoom.
- Fit to bounds.

Acceptance:
- Test cube rotates/pans/zooms smoothly.
- Zoom does not alter object coordinates.

### Phase 3: Atom Batch
Goal:
- Draw all atoms in one instanced draw call.

Files:
- `src/atomstudio/preview/gl/mesh.py`
- `src/atomstudio/preview/gl/batches.py`
- `src/atomstudio/preview/gl/shaders/atom.vert`
- `src/atomstudio/preview/gl/shaders/atom.frag`

Tasks:
- Generate sphere mesh.
- Upload atom instance buffer.
- Draw atom instances.
- Apply simple lighting.
- Apply per-atom color/radius.

Acceptance:
- `water.xyz` displays clean atoms.
- A 200-atom structure rotates smoothly.
- No bond rendering in this phase.

### Phase 4: Bond Batch
Goal:
- Draw all bonds in one instanced draw call without z-fighting.

Files:
- `src/atomstudio/preview/gl/shaders/bond.vert`
- `src/atomstudio/preview/gl/shaders/bond.frag`
- `src/atomstudio/preview/gl/batches.py`

Tasks:
- Generate open cylinder/tube mesh.
- Upload bond instance buffer.
- Compute transform from start/end/radius.
- Implement bicolor interpolation in shader.
- Support bond order by multiple instances.

Acceptance:
- Bonds render cleanly with no midpoint cap z-fighting.
- Adding bonds does not create many draw calls.
- A 200-atom bonded structure remains smooth.

### Phase 5: Picking and Selection
Goal:
- Restore atom/bond selection with correct depth priority.

Files:
- `src/atomstudio/preview/picking.py`
- `src/atomstudio/preview/selection.py`
- `src/atomstudio/preview/gl/shaders/selection.*`

Tasks:
- CPU ray-sphere picking.
- CPU ray-bond picking.
- Single select, multi-select, box select.
- Selected object payload for inspector.
- Depth-aware selection shell.

Acceptance:
- Double click in rotate mode selects correct front atom.
- Select mode single click selects atom/bond.
- Drag selection selects atoms and bonds.
- Selection highlight does not show through occluding atoms.

### Phase 6: App Integration
Goal:
- Replace VisPy preview as default backend.

Files:
- `src/atomstudio/app/window.py`
- `src/atomstudio/app/preview_widget.py`
- `src/atomstudio/cli.py`

Tasks:
- Add preview backend selection config if needed.
- Wire toolbar/menu actions to new controller.
- Inspector/output continue to work.
- Keep final Blender render path unchanged.

Acceptance:
- `atomstudio app --input tests/data/water.xyz` displays water with OpenGL backend by default.
- Existing app panels remain functional.
- Blender final render is not affected.

### Phase 7: Remove VisPy Mainline
Goal:
- Retire the current `preview_canvas.py` path after OpenGL backend is stable.

Tasks:
- Remove VisPy-specific renderer path or keep as separate deprecated module temporarily.
- Delete obsolete tests tied to VisPy visual internals.
- Keep preview data/model tests.

Acceptance:
- No app code imports `vispy` for core preview.
- Unit tests and GUI smoke tests pass.

## Recommended Subagent Split

### Agent A: Qt/OpenGL Shell
Write scope:
- `app/preview_widget.py`
- `preview/gl/renderer.py`
- `preview/gl/state.py`

Responsibilities:
- QOpenGLWidget lifecycle.
- OpenGL initialization.
- Resize/paint loop.
- Context diagnostics.

### Agent B: Camera and Interaction
Write scope:
- `preview/camera.py`
- mouse event bridge in `app/preview_widget.py`

Responsibilities:
- Orbit/pan/zoom.
- Fit/top/front/side/a/b/c views.
- Orthographic/perspective.

### Agent C: Atom Renderer
Write scope:
- `preview/gl/mesh.py`
- `preview/gl/batches.py`
- atom shaders

Responsibilities:
- Sphere mesh.
- Atom instance buffer.
- Atom shader.
- Atom draw tests.

### Agent D: Bond Renderer
Write scope:
- `preview/gl/batches.py`
- bond shaders

Responsibilities:
- Open cylinder/tube mesh.
- Bond instance buffer.
- Bicolor shader.
- Bond order.
- Performance tests for bonded structures.

### Agent E: Picking and Selection
Write scope:
- `preview/picking.py`
- `preview/selection.py`
- selection shaders/batch

Responsibilities:
- Ray picking.
- Box selection.
- Correct depth priority.
- Selection highlight.

### Agent F: App Integration and Tests
Write scope:
- `app/window.py`
- `cli.py`
- tests

Responsibilities:
- Default backend switch.
- Toolbar/menu integration.
- GUI smoke tests.
- Regression tests.

## Test Plan

### Unit Tests
- Camera matrix tests.
- Fit-to-bounds tests.
- Ray-sphere picking tests.
- Ray-bond picking tests.
- Atom/bond batch payload shape tests.
- Shader file existence/compile smoke where context is available.

### GUI Smoke Tests
- Create `QOpenGLWidget`.
- Initialize renderer.
- Load `water.xyz`.
- Load a bonded 200-atom structure fixture.
- Verify draw call counters are within expected range.

### Manual Acceptance
- Water molecule looks clean with atoms and bonds.
- No visible bond noise/z-fighting.
- Rotation is smooth.
- Zoom does not change atom spacing.
- Selection chooses front atom when atoms overlap in screen space.
- Bond selection works.

## Risks

### WSL / GWSL / WSLg Differences
OpenGL context creation and cursor behavior can differ. The backend must report:
- Qt platform
- OpenGL version
- renderer/vendor
- depth bits
- MSAA samples

### PyOpenGL Overhead
Acceptable if draw calls are batched. Avoid per-object calls.

### Shader Portability
Use conservative GLSL version first. Avoid advanced features until the baseline is stable.

### Transparent Objects
Defer transparency. Transparent sorting is a separate subsystem.

## Decision
AtomStudio should move to `QOpenGLWidget + self-owned OpenGL renderer` as the long-term preview architecture.

VisPy should not receive new feature work except short-term bug fixes required to keep the app usable during migration.
