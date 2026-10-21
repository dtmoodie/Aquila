# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Aquila is a node-based (dataflow graph) image/video processing framework for CPU+GPU pipelines. It sits in the stack **rcc → metaobject → aquila** (higher-level apps like eagleyeye build on aquila):

- **RCC** (Runtime Compiled C++) — `RuntimeObjectSystem`/`RuntimeCompiler`: classes are registered into a runtime type system and can be hot-recompiled from source. `rcc::shared_ptr` is used throughout instead of `std::shared_ptr` for managed objects.
- **MetaObject** (`dependencies/MetaObject`, submodule) — reflection layer on top of RCC: params, publishers/subscribers, signals/slots, serialization, fiber-based async streams. See its own `CLAUDE.md` for that project's details.
- **Aquila** (this repo) — the processing layer: `Algorithm` → `Node` → `Graph`, frame grabbers, and image/detection types.

Active development happens on the `devel` branch (PRs target `master`). Recent direction (see `git log`): ECS rework for object tracking, object-detection classification, new minitensor API.

## Build

```bash
git submodule update --init --recursive   # REQUIRED — submodules are often not checked out (see Gotchas)
mkdir build && cd build
cmake .. && make -j$(nproc)
```

- C++11 baseline (modules may bump `CXX_STANDARD`); CUDA is optional — when found (`MO_HAVE_CUDA`), modules with `.cu` files build via `cuda_add_library`. Requires Boost (system, filesystem, thread, date_time, iostreams, chrono, regex, log, atomic, unit_test_framework), OpenCV, Eigen3 (types module), optionally Python dev.
- **CUDA 13 / CCCL**: `.cu` test targets compile with `-std=c++17` (CUDA 13's Thrust/CUB/libcu++ headers hard-require C++17; the `CCCL_IGNORE_DEPRECATED_CPP_DIALECT` escape hatch does *not* work — the headers use real C++17 syntax). CCCL headers also live in `<cuda-root>/targets/<arch>/include/cccl`, not `<cuda-root>/include` — the top-level CMakeLists adds that dir for host code (e.g. `nodes/Node.cpp` includes `thrust/system/system_error.h`); nvcc finds it on its own.
- Test plugins are declared with `metaobject_declare_plugin(<tgt> NOINSTALL)` — the bare macro form runs `install(TARGETS ... EXPORT ...)` and fails, because the macro adds the caller's `src/` dir as a bare (non-`$<BUILD_INTERFACE>`) interface include, which CMake rejects in the install context.
- All executables and shared libs land in the **build root** (`CMAKE_BINARY_DIR`) — except the cache-engine/ctext test executables, which stay in their own sub-build dirs (`build/dependencies/cache-engine/`, `build/dependencies/ctext/tests/`): the top-level CMakeLists lifts the runtime-output redirect while adding those two, because their plain-name `add_test` entries are resolved by ctest relative to each test's own build dir (MetaObject and Aquila's own tests reference the build root by absolute path instead). Debug libs get a `d` suffix (`CMAKE_DEBUG_POSTFIX`).
- CMake options (defined mostly in MetaObject's CMakeLists, shared scope): `BUILD_TESTS` (ON), `WITH_PYTHON` (ON), `WITH_CUDA` (ON), `BUILD_DEPENDENCIES` (ON; OFF switches to `find_package(RCC)` etc.).
- Modules are declared with `aquila_declare_module()` (`cmake/aquila_declare_module.cmake`): creates shared lib `aquila_<name>`, globs `modules/<name>/src/**/*.{cpp,cu}`, installs `src/**/*.h(pp)` to `include/`. Sources live under `modules/<name>/src/Aquila/...` and are included as `<Aquila/...>`. A module with no `.cpp` sources becomes an INTERFACE (headers-only) target.
- `modules/gui` and `modules/rcc` are added explicitly in the top-level CMakeLists; every other `modules/*` directory is auto-discovered via `SUBDIRLIST`.
- Downstream consumers use the generated `AquilaConfig.cmake` (`find_package(Aquila)` → `Aquila_LIBS`).
- The devcontainer image (`dtmoodie:dev`, see `.devcontainer/`) has the full dependency stack preinstalled.

## Tests

googletest per module; test targets are named `test_aq_<subdir>` and land in the build root:

```bash
cd build
ctest                                   # whole suite
ctest -R test_aq_node                   # one target
./test_aq_node --gtest_filter='*'       # directly, with gtest filter
```

Targets: `test_aq_algorithm`, `test_aq_frame_grabbers`, `test_aq_node`, `test_aq_node_graph` (core), `test_aq_types` (includes a `.cu` test; also builds the `aq_cv_compression` plugin when OpenCV `cudaarithm`/`imgcodecs` are found), `test_aq_utilities`; gui tests (`gui_click_callback`, `scalar_plot`) are separate targets.

- **Run tests from the build root** when executing the binaries directly: plugins are loaded as `.so` files from there, and test data is copied there (e.g. types' `data/data.png` is `file(COPY)`'d into `CMAKE_BINARY_DIR`). The `add_test` entries register the binary by absolute path and set `WORKING_DIRECTORY` to the build root, so `ctest` and direct execution behave the same.
- Plugin tests exercise RCC's plugin machinery: `aq_cv_compression` is a shared lib registered via `metaobject_declare_plugin` and loaded at runtime by name.
- Python smoke test: `aquila.py` (wrapper generated from `aquila.py.in.unix`) and `aquila_python_test.py` are configured into the build root; `python aquila_python_test.py` (needs camera/OpenNI2 hardware for its sample graph).
- **The top-level `tests/` directory is dead code** — the top-level CMakeLists never adds it (it targets the old Boost.Test setup with `MetaObject`/`Aquila` lib names that no longer exist). Module tests live in `modules/<name>/tests/`.

## Architecture

### core (`modules/core`) — the graph engine

- **`aq::Algorithm`** (`core/Algorithm.hpp`) — base class for every processing unit. Wraps MetaObject's object system: inputs are `mo::ISubscriber`s, outputs `mo::IPublisher`s (declared via `INPUT`/`OUTPUT`/`SOURCE`/`PARAM` macros), plus signals/slots. Per-frame hook is `processImpl()`; the base implements timestamp/frame-number synchronization across multiple inputs (`SyncMethod`, `ParameterSynchronizer`, `checkInputs`).
- **`aq::nodes::Node`** (`nodes/Node.hpp`) — an `Algorithm` wrapped as a named node in a tree (parent/child operators on `INode`). Nodes are registered by name in the RCC runtime type system; `NodeFactory` (singleton) and `INode::listConstructableNodes()` discover them at runtime.
- **`aq::Graph`** (`core/Graph.hpp`) — the running stream. Owns nodes, connects publishers→subscribers by name, spins threads (`start`/`stop`), exposes the param server and relay manager for runtime introspection/updates, `saveGraph`/`loadDocument` for persistence, variable sinks for UI.
- **Frame grabbers** (`framegrabbers/`) — input sources (camera, file); `GrabberInfo::canLoad(path)` returns a priority so the right grabber is picked per data source.
- **Node authoring pattern** (see `modules/core/tests/node/objects.hpp` for the minimal example):

```cpp
struct MyNode : public aq::nodes::Node
{
    bool processImpl() override;
    MO_DERIVE(MyNode, Node)
        PARAM(int, threshold, 5)     // runtime-writable parameter
        INPUT(int, value)            // subscriber
        OUTPUT(int, result)          // publisher (SOURCE = publisher flagged as a source)
    MO_END;
};
```

### types (`modules/types`) — data plane

Image/detection types: `TSyncedMemory`/`SyncedMemory` (host<->device synced buffers), `SyncedImage`/`SyncedView` (wrap `cv::Mat` natively, including strided/padded data), `CompressedImage` + `IImageCompressor` plugin interface, `ObjectDetection`/`DetectionPatch` + `detection/`, `geometry/`, and `EntityComponentSystem` (object tracking, recently reworked). Tensor layer is **minitensor** (a `ctext` dependency) — `design/tensors.md` records the library evaluation that led there.

### other modules

- **rcc** — Aquila's glue to the RCC runtime compiler (`RUNTIME_MODIFIABLE_INCLUDE`/`RUNTIME_COMPILER_SOURCEDEPENDENCY` in core headers mark runtime-recompilable classes). `src/Aquila/rcc/external_includes/` holds shim headers (OpenCV, caffe) the runtime compiler includes when recompiling user code.
- **serialization** — cereal-based node/param serialization (`cereal/`), plus ROS message type helpers (`ros/`).
- **python** — Boost.Python bindings (built only when the `metaobject_python` target exists). Exposes `aquila.Graph`, `aquila.nodes.*`, `aquila.framegrabbers.*`; node constructors in Python take publisher tuples, e.g. `aquila.nodes.OGLImageDisplay(image=(node, node.getOutput('x')))`.
- **gui** — OpenCV highgui-based displays/parameter widgets (only when `opencv_highgui` found).
- **imgproc** — colormappers (CPU/GPU) and GPU drawing.
- **utilities** — thrust/CUDA interop, sorting, containers.

### execution model

MetaObject runs the graph on a **fiber-based `AsyncStream`** (boost.fiber). nvcc cannot compile boost/fiber headers — **keep `.cu` files free of fiber includes** (this is why `IAsyncStream` indirection exists in MetaObject).

## Gotchas

- **Submodules use relative URLs** (`.gitmodules`: `../MetaObject`, `../cache-engine`, `../ctext`), which resolve against the GitHub origin (`dtmoodie/` forks) — `git submodule update --init` clones the forks, not sibling local checkouts. To build against local checkouts, override first: `git config submodule.dependencies/<name>.url <abs path>` (per-repo config; nested submodules get the override in their own module config, e.g. `git -C dependencies/cache-engine config submodule.external/ct.url <path>`).
- **This container has no SSH key** (`ssh -T git@github.com` → `Permission denied (publickey)`), so submodule clones from the ssh-resolved URLs fail; HTTPS to GitHub works. The cache-engine nested chain (`external/ct` → `dependencies/minitensor` → `googletest`) was resolved in this checkout from local checkouts (MetaObject's `ct`, ctext's `minitensor`/`googletest`) via `submodule.<name>.url` overrides — all pins were already present locally. Clones from local *paths* additionally need `git -c protocol.file.allow=always submodule update ...` (git's file-transport hardening blocks submodule-initiated local clones otherwise).
- A failed checkout can leave a submodule's worktree empty with the deletions **staged in its index** while `git submodule status` still looks clean — restore with `git -C <submodule> reset --hard HEAD` (objects are local, no network needed).
- **`WITH_PYTHON=ON` needs CUDA-enabled OpenCV**: the MetaObject python module unconditionally includes `<opencv2/core/cuda.hpp>`, so CPU-only OpenCV configures fine then fails to compile. With CPU OpenCV, set `WITH_PYTHON=OFF`.
- **Run built tests from the build root** (working directory is used for plugin `.so` lookup and test data paths).
- The top-level `tests/` dir is not part of the build (legacy).
- `RCC_TARGET_CONFIG` (from RCC's cmake) wires per-config binary dirs so runtime-compiled plugins can find their libs at runtime; don't relocate build outputs without it.
- `.clang-format` (LLVM-based, 4-space indent, braces on own lines for namespaces/classes) defines the project style — format code with it.
- `dependencies/MetaObject/CLAUDE.md` documents the reflection macro DSL (`MO_DERIVE`, `PARAM`, `INPUT`, `OUTPUT`, `MO_SIGNAL`/`MO_SLOT`, `MO_END`) in depth — read it before touching MetaObject macros or the params module.
