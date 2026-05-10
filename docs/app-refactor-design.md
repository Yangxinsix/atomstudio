# AtomStudio App 架构重构设计

## Status
- Draft
- Date: 2026-04-27
- Owner: AtomStudio core

## Naming
- 统一使用 `app` 指代桌面界面层，不再使用 `gui`。
- `preview` 指代实时预览渲染层，不等同于 `app`。
- `backend` 指代高质量离线渲染后端，当前实现是 Blender backend。

## Executive Summary
当前项目的核心问题不是功能不够，而是职责边界混乱：
- `preview` 在重复实现一套场景理解逻辑。
- `scene` 同时承担“语义解析”和“Blender 对象创建”。
- `Structure` 直接耦合渲染调用。
- `app/preview_canvas.py` 同时承担数据转换、选择、相机、灯光、mesh 创建和 UI 事件，已经明显过载。

这次重构的目标是建立一个唯一的中间层：`RenderScene`。

今后所有渲染路径都必须遵守同一条链路：

1. `Structure + RenderJobConfig`
2. `SceneBuilder.build(...)`
3. `RenderScene`
4. `PreviewRendererVisPy.render(scene)` 或 `BlenderBackend.render(scene)`

这意味着：
- `structure / material / camera / light` 只描述属性，不直接创建后端对象。
- `scene` 只做后端无关的场景构建。
- `preview` 只负责把 `RenderScene` 变成 VisPy 可渲染对象。
- `backend/blender` 只负责把 `RenderScene` 变成 Blender 可渲染对象。
- `app` 只负责界面和交互，不负责场景语义。

## Goals
- 建立统一的后端无关场景描述层。
- 消除 preview 和 Blender 各自维护一套场景解析逻辑的问题。
- 把相机、灯光、材质、几何的语义从具体后端实现中拆出来。
- 让 `app` 成为纯 UI 层。
- 为后续增加更多 backend 或 preview 特性保留清晰扩展点。

## Non-Goals
- 本轮不追求 Blender 和 VisPy 完全一致的像素级结果。
- 本轮不重写配置格式。
- 本轮不引入第三套渲染后端。
- 本轮不优先做新功能，先收敛架构。

## Current Problems

### 1. Preview 和 Blender 场景语义分裂
- `src/atomstudio/preview/builder.py` 直接从 `Structure + RenderJobConfig` 构建 preview buffers。
- `src/atomstudio/scene/composer.py` 再独立走一套 Blender scene build 流程。
- 两边共享的是零散函数，不是统一的场景模型。

结果：
- 新增一个样式规则时，preview 和 Blender 都要改。
- 两边很容易出现表示、半径、颜色、split bond、polyhedra 等行为漂移。

### 2. Scene 层掺杂 Blender 专属实现
- `src/atomstudio/scene/structure_renderer.py` 既在做样式解析，又直接创建 Blender 对象。
- `src/atomstudio/scene/camera_builder.py` 和 `src/atomstudio/scene/lights/builder.py` 同时包含语义解析和 `bpy` 创建逻辑。

结果：
- preview 无法直接复用 scene 层。
- 任何非 Blender 路径都只能复制逻辑。

### 3. Structure 模型直接调用渲染
- `src/atomstudio/structure/structure.py` 中 `Structure.get_image()` 直接走 render pipeline。

结果：
- 领域模型和输出后端耦合。
- `Structure` 不再是纯数据/操作对象。

### 4. App 和 Preview 边界模糊
- `src/atomstudio/app/preview_canvas.py` 当前承担了过多职责：
  - 数据转换
  - mesh 构建
  - picking
  - selection payload
  - axis overlay
  - 相机数学
  - fallback 策略
  - Qt 事件桥接

结果：
- 文件持续膨胀。
- 测试边界不清晰。
- 任何 UI 改动都容易污染渲染逻辑。

## Target Architecture

### Layer Model

#### 1. Domain Layer
负责描述对象本身，不关心 Blender 或 VisPy。

Modules:
- `structure/`
- `scene/materials/specs.py`
- `scene/lights/specs.py`
- `config/`

Responsibilities:
- 原子、键、晶胞、多面体的原始数据
- 材质参数
- 灯光参数
- 相机参数
- 配置定义

#### 2. Scene Layer
负责把 domain 数据和配置解析成统一场景。

Modules:
- `scene/model.py`
- `scene/builder.py`
- `scene/styling.py`
- `scene/geometry.py`
- `scene/transforms.py`
- `scene/camera_resolver.py`
- `scene/light_resolver.py`

Responsibilities:
- 表示法解析
- 半径解析
- 颜色解析
- 材质策略解析
- bond split 语义
- boundary expansion
- model rotation
- polyhedra triangulation
- camera pose 解析
- light runtime spec 解析

Output:
- `RenderScene`

#### 3. Preview Layer
负责把统一场景翻译成 VisPy 渲染对象。

Modules:
- `preview/renderer.py`
- `preview/material_adapter.py`
- `preview/mesh_builder.py`
- `preview/picking.py`
- `preview/selection.py`

Responsibilities:
- `RenderScene -> VisPy meshes`
- preview 专属材质近似
- picking 和 selection 命中检测
- preview fallback

禁止事项:
- 不允许重新解析结构语义
- 不允许自行决定 bond/atom 的表示逻辑
- 不允许直接操作 app 状态

#### 4. Backend Layer
负责把统一场景翻译成高质量后端对象。

Modules:
- `backend/blender/renderer.py`
- `backend/blender/material_adapter.py`
- `backend/blender/camera_writer.py`
- `backend/blender/light_writer.py`
- `backend/blender/scene_writer.py`

Responsibilities:
- `RenderScene -> bpy objects`
- Blender 材质节点创建
- Blender 相机和灯光落地
- 最终离线渲染

#### 5. App Layer
负责桌面界面和交互。

Modules:
- `app/main.py`
- `app/window.py`
- `app/menus.py`
- `app/state.py`
- `app/inspector.py`
- `app/controllers/`
- `app/widgets/`

Responsibilities:
- 窗口、菜单、dock、状态栏
- 用户输入
- 文件加载
- 任务调度
- 选择同步
- 调用 preview renderer
- 调用 backend render worker

禁止事项:
- 不允许自己解析结构语义
- 不允许直接决定材质/灯光/相机规则
- 不允许直接持有 Blender 专属对象

## Proposed File Layout

```text
src/atomstudio/
  app/
    main.py
    window.py
    menus.py
    state.py
    inspector.py
    controllers/
      preview_controller.py
      render_controller.py
    widgets/
      preview_view.py
      axis_overlay.py

  scene/
    model.py
    builder.py
    styling.py
    geometry.py
    transforms.py
    camera_resolver.py
    light_resolver.py
    materials/
      specs.py
      registry.py
      request.py
    lights/
      specs.py

  preview/
    renderer.py
    material_adapter.py
    mesh_builder.py
    picking.py
    selection.py

  backend/
    blender/
      renderer.py
      material_adapter.py
      camera_writer.py
      light_writer.py
      scene_writer.py

  structure/
    structure.py
    atom.py
    bond.py
    cell.py
    polyhedron.py
```

## Core Data Contract

### RenderScene
`RenderScene` 是整个系统的唯一场景真相源。

建议字段：
- `atoms: list[SceneAtom]`
- `bonds: list[SceneBond]`
- `polyhedra: list[ScenePolyhedron]`
- `cell_edges: list[SceneCellEdge]`
- `camera: SceneCamera`
- `lights: list[SceneLight]`
- `background: tuple[float, float, float, float]`
- `metadata: dict[str, Any]`
- `report: dict[str, Any]`

### SceneAtom
- `index`
- `symbol`
- `atomic_number`
- `position`
- `radius`
- `representation`
- `material`
- `selection_payload`
- `metadata`

### SceneBond
- `id`
- `a`
- `b`
- `order`
- `bond_type`
- `distance`
- `radius`
- `segments`
- `material_uniform`
- `material_left`
- `material_right`
- `selection_payload`
- `metadata`

### SceneCamera
- `projection`
- `center`
- `right`
- `up`
- `forward`
- `scale_factor`
- `lens_mm`
- `clip_start`
- `clip_end`
- `metadata`

### SceneLight
- `type`
- `location`
- `direction`
- `energy`
- `size`
- `color`
- `lock_to_camera`
- `metadata`

## Refactor Rules

### Rule 1
任何 preview/backend 代码都不得直接从 `Structure + Config` 推断表示法、半径、灯光、相机。

### Rule 2
`scene` 层不得创建 `bpy` 对象，也不得引用 Qt/VisPy 类型。

### Rule 3
`app` 层不得直接 import Blender 专属逻辑。

### Rule 4
`Structure.get_image()` 是保留的便利 API，但必须只委托标准 render pipeline；不得在 `structure` 层实现后端细节。

### Rule 5
所有选择信息都应该附着在 `RenderScene` 节点上，而不是由 UI 现算业务字段。

## Migration Plan

### Phase 0: Freeze
目标：
- 停止继续往 `app/preview_canvas.py` 和 `scene/structure_renderer.py` 塞新语义逻辑。

动作：
- 新增架构约束说明。
- 明确所有新功能必须通过中间层设计评审。

### Phase 1: Introduce RenderScene
目标：
- 建立统一的中间数据模型。

动作：
- 新建 `scene/model.py`
- 定义 `RenderScene`、`SceneAtom`、`SceneBond`、`SceneCamera`、`SceneLight`
- 保持现有 preview/blender 路径暂时不改输出

完成标准：
- `RenderScene` 数据类稳定
- 单测覆盖序列化、字段完整性、selection payload

### Phase 2: Extract SceneBuilder
目标：
- 把场景语义从 preview/blender 里抽出来。

动作：
- 新建 `scene/builder.py`
- 新建 `scene/styling.py`
- 新建 `scene/transforms.py`
- 从 `scene/structure_renderer.py` 抽出：
  - 表示法解析
  - 半径解析
  - atom/bond style 解析
  - material policy 解析
  - bond split 规则
- 从 `scene/composer.py` 抽出：
  - boundary expansion
  - model rotation
- 从 `preview/builder.py` 移除独立语义解析

完成标准：
- `SceneBuilder.build(structure, cfg)` 可直接产出 `RenderScene`
- preview 和 backend 不再直接做语义解析

### Phase 3: Migrate Blender Backend
目标：
- Blender 只消费统一场景。

动作：
- 新建 `backend/blender/renderer.py`
- 新建 `backend/blender/scene_writer.py`
- 新建 `backend/blender/material_adapter.py`
- 将 `scene/composer.py` 和 `scene/structure_renderer.py` 迁移为 backend 层适配器

完成标准：
- `render/pipeline.py` 改为 `Structure + Config -> RenderScene -> BlenderBackend`
- Blender 输出结果与当前行为一致

### Phase 4: Migrate Preview
目标：
- preview 只消费统一场景。

动作：
- 新建 `preview/renderer.py`
- 新建 `preview/mesh_builder.py`
- 新建 `preview/material_adapter.py`
- 新建 `preview/picking.py`
- 将 `app/preview_canvas.py` 缩减为 Qt widget 包装层

完成标准：
- preview 不再 import `scene.composer` 私有逻辑
- preview 不再拥有独立 scene build 规则

### Phase 5: Clean App Layer
目标：
- app 只保留界面职责。

动作：
- `app/preview_canvas.py` 拆成 `app/widgets/preview_view.py`
- 把 selection、controller、task orchestration 分开
- `window.py` 只保留布局和 action wiring

完成标准：
- `window.py` 不再直接构造 selection payload
- `app` 只调用 preview/backend 接口

### Phase 6: Remove Legacy Paths
目标：
- 删除重复逻辑和历史兼容层。

动作：
- 删除或薄封装：
  - `preview/builder.py`
  - `scene/composer.py` 中不再需要的逻辑
  - `scene/structure_renderer.py` 中语义解析部分
  - `Structure.get_image()` 保留为便利 API，但只允许委托标准 render pipeline

完成标准：
- 所有渲染路径统一走 `RenderScene`

## Work Split

### Workstream A: Scene Core
Owner:
- Agent A

Scope:
- `scene/model.py`
- `scene/builder.py`
- `scene/styling.py`
- `scene/transforms.py`
- `scene/camera_resolver.py`
- `scene/light_resolver.py`

Responsibilities:
- 建立 `RenderScene`
- 统一表示法、半径、材质、灯光、相机解析
- 抽离 preview 和 Blender 共用语义

Do not touch:
- `app/`
- `backend/blender/`

### Workstream B: Blender Backend
Owner:
- Agent B

Scope:
- `backend/blender/*`
- `render/pipeline.py`

Responsibilities:
- `RenderScene -> bpy`
- 材质适配
- 相机适配
- 灯光适配
- 最终渲染保持兼容

Do not touch:
- `preview/`
- `app/`

### Workstream C: Preview Renderer
Owner:
- Agent C

Scope:
- `preview/renderer.py`
- `preview/mesh_builder.py`
- `preview/material_adapter.py`
- `preview/picking.py`

Responsibilities:
- `RenderScene -> VisPy`
- mesh 构建
- picking
- selection target
- fallback 策略

Do not touch:
- `backend/blender/`
- `app/window.py`

### Workstream D: App Shell
Owner:
- Agent D

Scope:
- `app/main.py`
- `app/window.py`
- `app/menus.py`
- `app/state.py`
- `app/inspector.py`
- `app/controllers/*`
- `app/widgets/*`

Responsibilities:
- 菜单
- dock
- 控制器
- 任务编排
- app 状态管理

Do not touch:
- scene 语义解析
- backend 材质/灯光/相机适配

### Workstream E: Migration and QA
Owner:
- Agent E

Scope:
- `tests/unit/*`
- `tests/integration/*`
- 临时兼容层

Responsibilities:
- 测试迁移
- 回归验证
- 删除旧路径前的兼容验证

## Immediate File Moves

### Extract from `scene/structure_renderer.py`
移动到 `scene/styling.py`：
- `resolve_representation`
- `resolve_draw_bonds_with_atom_representations`
- `resolve_atom_style_state`
- 半径计算相关纯函数
- bond split 颜色/材质解析逻辑

保留在 Blender backend：
- `Atom.build(...)` / `Bond.build(...)` / Blender collection/object 创建

### Extract from `scene/composer.py`
移动到 `scene/transforms.py`：
- `_apply_model_rotation`
- `_resolve_model_rotation_matrix`
- 旋转中心和点/向量变换

保留在 backend：
- `clear_scene`
- `bpy.context.scene[...]`
- 最终 render 调用

### Extract from `app/preview_canvas.py`
移动到 `preview/picking.py`：
- 屏幕投影
- atom hit-test
- bond hit-test
- tie-break 规则

移动到 `preview/mesh_builder.py`：
- sphere mesh 生成
- cylinder mesh 生成
- split/double/triple bond 展开

保留在 app：
- Qt widget 包装
- 事件桥接
- axis overlay widget 装配

## Acceptance Criteria

### Architecture
- preview 和 backend 不再各自持有一套场景构建逻辑。
- `RenderScene` 成为唯一场景真相源。
- `app` 不再直接构造渲染语义。

### Code Health
- `app/preview_canvas.py` 缩减为 UI 包装层。
- `scene/structure_renderer.py` 不再承担 Blender 之外的场景语义解析。
- `Structure.get_image()` 只作为便利包装存在，内部走标准 Blender render pipeline。

### Tests
- `SceneBuilder` 有独立单测。
- preview renderer 和 backend renderer 各自有 adapter 单测。
- 原子、键、polyhedra、cell 的语义在 preview/backend 下输出一致。
- 相机和灯光解析在 preview/backend 下来源一致。

## Risks
- 初期会有较多兼容层，短时间内文件数会上升。
- 如果边拆边继续加功能，重构会失败。
- Blender 输出和 preview 输出的视觉差异在过渡期仍然存在，但语义必须先统一。

## Recommendation
- 先暂停继续扩展 preview 功能。
- 第一优先级不是“再做一个功能”，而是“做出 `RenderScene` 并让 Blender 先接上”。
- 只有当 Blender 和 preview 都改成消费统一场景后，再继续加材质细节、交互和高级菜单。
