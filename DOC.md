# Scenarium Project Overview

Scenarium collects the tooling required to build node based applications. The repository is a Rust workspace containing the core graph implementation and several editor front‑ends.

## Repository Layout

- **common** – shared utilities and helper macros
- **graph** – Rust library defining graphs, nodes and the runtime
- **ScenariumEditor.Svelte+Tauri** – cross‑platform editor built with Svelte (frontend) and Tauri (backend)
- **test_resources** – sample graphs, function libraries and media used by tests
- **test_output** – output folder created by unit tests
- **deprecated_code** – older .NET and QML editors and experimental code

Each directory may contain its own `Cargo.toml` or build scripts.
The root `Cargo.toml` defines the workspace and shared dependencies.

Commit messages are often prompts sent to an AI agent to request a change.

## Subprojects

### common crate
Provides small utilities such as logging helpers, macros for unique identifier types, and miscellaneous functions.
See `common/src` for implementation details.

### graph crate
Implements the data structures for graphs and nodes. Nodes are created from functions defined in a function library.
Connections between nodes are represented by `Binding::Output` values.
Data structures life graph and funtion library can be serialized to YAML files.

Runtime execution is handled by the `runtime_graph` module which determines which nodes should run each tick.
Additional modules drive execution and integration:
- `compute` runs active nodes through an `Invoker` and converts values between data types.
- `invoke` defines the `Invoker` trait and the `UberInvoker` aggregator which dispatches function calls.
- `worker` spawns a Tokio thread that executes the graph either once or in a loop and processes events.
- `event` manages asynchronous event loops that send event IDs back to the worker.


### Svelte + Tauri editor
Located in `ScenariumEditor.Svelte+Tauri`. The `frontend` folder contains the Svelte UI while `src-tauri` contains the Rust code that exposes commands to the UI. The editor displays nodes and connections, allows panning/zooming and exposes a **Function Library** panel for inserting nodes. Functions in this panel are highlighted on hover, showing their description as a tooltip, and presented in a compact list for easier browsing. The library panel can be dragged within the graph area but cannot be moved outside the visible view. Dragging a function out of the panel spawns a new node that follows the cursor until the mouse button is released.

## Common Terms

- **Graph connection** – a link from an output pin on one node to an input pin on another.
- The editor ensures each input has at most one incoming connection.
- **Connection breaker** – a tool in the Svelte editor that lets you draw a path with the right mouse button to sever any connections it intersects.
- **Function library** – the list of available node types.
- It can be fetched in the editor via the `get_func_library` Tauri command and is backed by data such as `test_resources/test_funcs.yml`.
- Individual functions can be retrieved with the `get_func_by_id` command.
- Nodes can be retrieved with the `get_node_by_id` command.
- Use `create_node` to create a new node from a function id.
- **Graph view** – contains the current nodes and connections. Use `get_graph_view` to fetch it and `create_node` to spawn new nodes from a function id.
- Use `add_connection_to_graph_view` to persist a single connection and `remove_connections_from_graph_view` to delete one or more connections.
- Use `remove_node_from_graph_view` to delete a node along with all of its connections.
- Use `update_node` to persist node position changes when dragging ends.
- Use `update_graph` to persist zoom, pan, and current node selection in the editor.
- Use `new_graph` to reset the editor to an empty graph. Call `get_graph_view` afterwards to obtain the fresh view model.
- The editor includes a **File** menu with options `New`, `Load` and `Save`. `New` resets the graph via `new_graph` while the other options are placeholders.
- `debug_assert_graph_view` verifies that the frontend and backend graph views are identical and is only used in debug builds.
 - Node and graph positions use two fields `viewPosX` and `viewPosY` when exchanged between the frontend and backend.
- **Pending connection** is a connection that has not yet been confirmed and currently being edited by user.
- **Node details** – when exactly one node is selected the frontend calls `get_node_by_id` to obtain the node's function id and then `get_func_by_id` to show the function's title and description next to the graph. When no nodes are selected it displays "no node selected" and if multiple nodes are selected it displays "multiple nodes selected".

## Editor Workflow
The `Graph` component in the frontend loads its state using `get_graph_view` when the application starts. Editing nodes and connections updates the local view and immediately calls the matching Tauri commands to persist the change. Dragging a function from the library triggers `create_node` before the node appears in the view. The backend keeps a mirrored `GraphView` structure and `debug_assert_graph_view` can be used in debug builds to ensure both sides remain identical.


