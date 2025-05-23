<script lang="ts">
    import Node from '$lib/Node.svelte';
    import FuncLibrary from '$lib/FuncLibrary.svelte';
    import type {GraphView, ConnectionView, Pin, NodeView, FuncView} from "$lib/types";

    interface GraphProps {
        selectedChange?: (node_ids: string[]) => void;
    }

    let {selectedChange}: GraphProps = $props();


    import {onMount} from 'svelte';
    import {invoke} from '@tauri-apps/api/core';


    const SNAP_RADIUS = 15;
    let moveHandler: (e: PointerEvent) => void;
    let upHandler: (e: PointerEvent) => void;

    const BG_DOT_BASE = 10;

    function mod(n: number, m: number) {
        return ((n % m) + m) % m;
    }

    const dotFactor = $derived(() => graphView.viewScale / Math.pow(2, Math.floor(Math.log2(graphView.viewScale))));
    const dotSpacing = $derived(() => BG_DOT_BASE * dotFactor());
    const bgX = $derived(() => mod(graphView.viewPosX, dotSpacing()));
    const bgY = $derived(() => mod(graphView.viewPosY, dotSpacing()));
    const zoomPercent = $derived(() => Math.round(graphView.viewScale * 100));

    const pendingPath = $derived(() => pendingConnection
        ? pendingConnection.start.type === 'output'
            ? pathBetween(
                getPinPos(pendingConnection.start),
                pendingConnection.hover ? getPinPos(pendingConnection.hover) : {
                    x: pendingConnection.x,
                    y: pendingConnection.y
                }
            )
            : pathBetween(
                pendingConnection.hover ? getPinPos(pendingConnection.hover) : {
                    x: pendingConnection.x,
                    y: pendingConnection.y
                },
                getPinPos(pendingConnection.start)
            )
        : '');
    const breakerPath = $derived(() => connectionBreaker
        ? connectionBreaker.points
            .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
            .join(' ')
        : '');
    const selectionBox = $derived(() => selection
        ? {
            left: Math.min(selection.startX, selection.x) - selection.rect.left,
            top: Math.min(selection.startY, selection.y) - selection.rect.top,
            width: Math.abs(selection.startX - selection.x),
            height: Math.abs(selection.startY - selection.y)
        }
        : null);

    let panning = false;
    let panStartX = 0;
    let panStartY = 0;
    let startViewX = 0;
    let startViewY = 0;
    let mainContainerEl: HTMLDivElement;

    let showFuncLibrary = $state(false);
    let selection: {
        startX: number;
        startY: number;
        x: number;
        y: number;
        pointerId: number;
        rect: DOMRect;
    } | null = $state(null);
    let connectionBreaker: {
        points: { x: number; y: number }[];
        pointerId: number;
    } | null = $state(null);

    let pendingConnection: { start: Pin; x: number; y: number; hover: Pin | null } | null = $state(null);

    // Track a node spawned from the function library that is being dragged
    let newNodeDrag: { nodeId: string; pointerId: number } | null = null;

    let graphView: GraphView = $state({
        nodes: [],
        connections: [],
        viewPosX: 0,
        viewPosY: 0,
        viewScale: 1,
        selectedNodeIds: new Set(),
    });

    function updateSelection() {
        selectedChange?.([...graphView.selectedNodeIds]);
        invoke('update_graph', {
            viewScale: graphView.viewScale,
            viewPosX: graphView.viewPosX,
            viewPosY: graphView.viewPosY,
        });
    }

    async function verifyGraphView() {
        if (import.meta.env.PROD) return;
        try {
            await invoke('debug_assert_graph_view', {
                graphView: {
                    nodes: graphView.nodes,
                    connections: graphView.connections,
                    viewScale: graphView.viewScale,
                    viewPosX: graphView.viewPosX,
                    viewPosY: graphView.viewPosY,
                    selectedNodeIds: [...graphView.selectedNodeIds]
                }
            });
        } catch (e) {
            console.error('Graph view mismatch', e);
        }
    }


    onMount(async () => {
        try {
            const data: GraphView = await invoke('get_graph_view');
            graphView.nodes = data.nodes;
            graphView.selectedNodeIds = new Set(data.selectedNodeIds);
            graphView.connections = [...data.connections];
            graphView.viewPosX = data.viewPosX;
            graphView.viewPosY = data.viewPosY;
            graphView.viewScale = data.viewScale;

            updateSelection();
            await verifyGraphView();

            console.log('Graph data loaded:', data);
        } catch (e) {
            console.error('Failed to load graph data', e);
        }
    });

    let connectionPaths: string[] = $state([]);
    let pendingConnectionPath: string = $state('');
    let trigger: number = $state(0);

    $effect(() => {
        trigger;
        graphView.nodes;
        graphView.connections;
        graphView.viewPosX;
        graphView.viewPosY;
        graphView.viewScale;

        connectionPaths = graphView.connections.map(connectionPath);
        pendingConnectionPath = pendingPath();
    });

    function queryPins() {
        return Array.from(
            mainContainerEl.querySelectorAll('[data-pin-type]')
        ) as HTMLElement[];
    }

    function findNearestPin(x: number, y: number, startType: 'input' | 'output', startPin?: Pin): Pin | null {
        let nearest: Pin | null = null;
        let nearestDist = Infinity;
        const rect = mainContainerEl.getBoundingClientRect();
        for (const el of queryPins()) {
            const pinType = el.getAttribute('data-pin-type') as 'input' | 'output';
            const nodeId = el.getAttribute('data-node-id');
            const indexStr = el.getAttribute('data-pin-index');
            if (!nodeId || !indexStr) continue;
            if (pinType === startType) continue;
            const index = Number(indexStr);
            if (
                startPin &&
                startPin.nodeId === nodeId &&
                startPin.type === pinType &&
                startPin.index === index
            ) {
                continue;
            }
            const r = el.getBoundingClientRect();
            const px = (r.left + r.width / 2 - rect.left - graphView.viewPosX) / graphView.viewScale;
            const py = (r.top + r.height / 2 - rect.top - graphView.viewPosY) / graphView.viewScale;
            const dx = px - x;
            const dy = py - y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < SNAP_RADIUS && dist < nearestDist) {
                nearestDist = dist;
                nearest = { nodeId, type: pinType, index };
            }
        }

        return nearest;
    }

    function onNodeSelect(detail: { nodeId: string; shiftKey: boolean }) {
        if (detail.shiftKey) {
            graphView.selectedNodeIds.add(detail.nodeId);
            graphView.selectedNodeIds = new Set(graphView.selectedNodeIds);
        } else {
            if (graphView.selectedNodeIds.has(detail.nodeId) && graphView.selectedNodeIds.size > 1) {
                return;
            }
            graphView.selectedNodeIds = new Set([detail.nodeId]);
        }
        updateSelection();
    }

    function dragNode(detail: { nodeId: string; dx: number; dy: number }) {
        const node = graphView.nodes.find((n) => n.id === detail.nodeId);
        if (!node) return;
        node.viewPosX += detail.dx;
        node.viewPosY += detail.dy;

        if (graphView.selectedNodeIds.has(detail.nodeId)) {
            for (const n of graphView.nodes) {
                if (n.id !== detail.nodeId && graphView.selectedNodeIds.has(n.id)) {
                    n.viewPosX += detail.dx;
                    n.viewPosY += detail.dy;
                }
            }
        }

        trigger++;
    }

    function endDragNode(nodeId: string) {
        const ids = graphView.selectedNodeIds.has(nodeId)
            ? [...graphView.selectedNodeIds]
            : [nodeId];
        for (const nid of ids) {
            const n = graphView.nodes.find((nn) => nn.id === nid);
            if (n) {
                invoke('update_node', {id: n.id, viewPosX: n.viewPosX, viewPosY: n.viewPosY});
            }
        }
    }

    function removeNode(nodeId: string) {
        graphView.nodes = graphView.nodes.filter((n) => n.id !== nodeId);
        graphView.connections = graphView.connections.filter(
            (c) => c.fromNodeId !== nodeId && c.toNodeId !== nodeId
        );

        graphView.selectedNodeIds.delete(nodeId);
        updateSelection();

        invoke('remove_node_from_graph_view', {id: nodeId})
            .then(() => verifyGraphView())
            .catch((e) => console.error('Failed to remove node', e));

    }

    async function startFuncDrag(item: FuncView, event: PointerEvent) {
        const rect = mainContainerEl.getBoundingClientRect();
        const x = (event.clientX - rect.left - graphView.viewPosX) / graphView.viewScale;
        const y = (event.clientY - rect.top - graphView.viewPosY) / graphView.viewScale;

        mainContainerEl.setPointerCapture(event.pointerId);
        try {
            // Persist the node before it becomes part of the view or is selected.
            const raw = await invoke<NodeView>('create_node', {funcId: item.id});
            const node: NodeView = {
                ...raw,
                viewPosX: x,
                viewPosY: y
            };

            graphView.nodes = [...graphView.nodes, node];
            graphView.selectedNodeIds = new Set([node.id]);
            newNodeDrag = {nodeId: node.id, pointerId: event.pointerId};
        } catch (e) {
            console.error('Failed to persist new node', e);
            mainContainerEl.releasePointerCapture(event.pointerId);
            return;
        }


        updateSelection();
        await verifyGraphView();

        showFuncLibrary = false;
    }

    function startConnection(detail: Pin & { x: number; y: number }) {
        const {nodeId, type, index, x, y} = detail;
        const rect = mainContainerEl.getBoundingClientRect();
        const nx = (x - rect.left - graphView.viewPosX) / graphView.viewScale;
        const ny = (y - rect.top - graphView.viewPosY) / graphView.viewScale;
        const startPin = {nodeId, type, index};
        pendingConnection = {
            start: startPin,
            x: nx,
            y: ny,
            hover: findNearestPin(nx, ny, type, startPin)
        };
        moveHandler = (e: PointerEvent) => {
            if (pendingConnection) {
                const mx = (e.clientX - rect.left - graphView.viewPosX) / graphView.viewScale;
                const my = (e.clientY - rect.top - graphView.viewPosY) / graphView.viewScale;
                pendingConnection.x = mx;
                pendingConnection.y = my;
                pendingConnection.hover = findNearestPin(mx, my, pendingConnection.start.type, pendingConnection.start);
            }
        };
        upHandler = () => {
            if (pendingConnection) {
                const target = pendingConnection.hover ?? pendingConnection.start;
                endConnection({
                    nodeId: target.nodeId,
                    type: target.type,
                    index: target.index
                });
            }
        };
        window.addEventListener('pointermove', moveHandler);
        window.addEventListener('pointerup', upHandler);
    }

    function endConnection(pin: Pin) {
        if (!pendingConnection) return;

        if (pendingConnection.start.nodeId === pin.nodeId && pendingConnection.start.type === pin.type && pendingConnection.start.index === pin.index) {
            pendingConnection = null;
            window.removeEventListener('pointermove', moveHandler);
            window.removeEventListener('pointerup', upHandler);
            return;
        }

        if (pendingConnection.start.type !== pin.type) {
            const from: Pin = pendingConnection.start.type === 'output'
                ? pendingConnection.start
                : {nodeId: pin.nodeId, type: 'output' as const, index: pin.index};
            const to: Pin = pendingConnection.start.type === 'input'
                ? pendingConnection.start
                : {nodeId: pin.nodeId, type: 'input' as const, index: pin.index};

            // ensure each input has at most one incoming connection
            graphView.connections = graphView.connections.filter(
                (c) => !(c.toNodeId === to.nodeId && c.toIndex === to.index)
            );

            let newConnection: ConnectionView = {
                fromNodeId: from.nodeId,
                fromIndex: from.index,
                toNodeId: to.nodeId,
                toIndex: to.index
            };

            graphView.connections = [
                ...graphView.connections,
                newConnection
            ];
            invoke('add_connection_to_graph_view', {connection: newConnection})
                .then(() => verifyGraphView())
                .catch((e) => {
                    console.error('Failed to persist new connection', e);
                });
        }

        pendingConnection = null;
        window.removeEventListener('pointermove', moveHandler);
        window.removeEventListener('pointerup', upHandler);
    }

    function getPinPos(pin: Pin) {
        const selector = `[data-node-id="${pin.nodeId}"] [data-pin-type="${pin.type}"][data-pin-index="${pin.index}"]`;
        const el = mainContainerEl.querySelector(selector) as HTMLElement | null;
        if (!el) return { x: 0, y: 0 };
        const rect = mainContainerEl.getBoundingClientRect();
        const r = el.getBoundingClientRect();
        return {
            x: (r.left + r.width / 2 - rect.left - graphView.viewPosX) / graphView.viewScale,
            y: (r.top + r.height / 2 - rect.top - graphView.viewPosY) / graphView.viewScale,
        };
    }

    function getInputPinPos(nodeId: string, index: number) {
        let inputPin: Pin = {nodeId: nodeId, type: 'input', index};
        return getPinPos(inputPin)
    }

    function getOutputPinPos(nodeId: string, index: number) {
        let outputPin: Pin = {nodeId: nodeId, type: 'output', index};
        return getPinPos(outputPin)
    }

    function pathBetween(p1: { x: number; y: number }, p2: { x: number; y: number }) {
        const dx = Math.abs(p2.x - p1.x);
        const offset = Math.max(dx / 2, 50);
        return `M ${p1.x} ${p1.y} C ${p1.x + offset} ${p1.y} ${p2.x - offset} ${p2.y} ${p2.x} ${p2.y}`;
    }

    function connectionSegments(
        p1: { x: number; y: number },
        p2: { x: number; y: number },
        segments = 20
    ) {
        const dx = Math.abs(p2.x - p1.x);
        const offset = Math.max(dx / 2, 50);
        const cp1 = {x: p1.x + offset, y: p1.y};
        const cp2 = {x: p2.x - offset, y: p2.y};
        const pts = [] as { x: number; y: number }[];
        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const mt = 1 - t;
            const x =
                mt * mt * mt * p1.x +
                3 * mt * mt * t * cp1.x +
                3 * mt * t * t * cp2.x +
                t * t * t * p2.x;
            const y =
                mt * mt * mt * p1.y +
                3 * mt * mt * t * cp1.y +
                3 * mt * t * t * cp2.y +
                t * t * t * p2.y;
            pts.push({x, y});
        }
        return pts;
    }

    function segmentsIntersect(p1: { x: number; y: number }, p2: { x: number; y: number }, q1: {
        x: number;
        y: number
    }, q2: { x: number; y: number }) {
        const orient = (a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }) =>
            (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        const o1 = orient(p1, p2, q1);
        const o2 = orient(p1, p2, q2);
        const o3 = orient(q1, q2, p1);
        const o4 = orient(q1, q2, p2);

        if (o1 === 0 && o2 === 0) {
            const minax = Math.min(p1.x, p2.x);
            const maxax = Math.max(p1.x, p2.x);
            const minay = Math.min(p1.y, p2.y);
            const maxay = Math.max(p1.y, p2.y);
            const minbx = Math.min(q1.x, q2.x);
            const maxbx = Math.max(q1.x, q2.x);
            const minby = Math.min(q1.y, q2.y);
            const maxby = Math.max(q1.y, q2.y);
            return minax <= maxbx && maxax >= minbx && minay <= maxby && maxay >= minby;
        }

        return o1 * o2 <= 0 && o3 * o4 <= 0;
    }

    function connectionPath(conn: ConnectionView) {
        return pathBetween(getOutputPinPos(conn.fromNodeId, conn.fromIndex), getInputPinPos(conn.toNodeId, conn.toIndex));
    }

    function onWheel(event: WheelEvent) {
        event.preventDefault();
        const rect = mainContainerEl.getBoundingClientRect();
        const mx = event.clientX - rect.left;
        const my = event.clientY - rect.top;
        const scaleFactor = event.deltaY < 0 ? 1.1 : 0.9;
        const newScale = Math.min(Math.max(graphView.viewScale * scaleFactor, 0.2), 4);
        const gx = (mx - graphView.viewPosX) / graphView.viewScale;
        const gy = (my - graphView.viewPosY) / graphView.viewScale;

        graphView.viewPosX = mx - gx * newScale;
        graphView.viewPosY = my - gy * newScale;
        graphView.viewScale = newScale;
        invoke('update_graph', {
            viewScale: graphView.viewScale,
            viewPosX: graphView.viewPosX,
            viewPosY: graphView.viewPosY
        });
    }

    function onContextMenu(event: MouseEvent) {
        if (connectionBreaker) {
            event.preventDefault();
        }
    }

    function cancelBreaker() {
        if (connectionBreaker) {
            mainContainerEl.releasePointerCapture(connectionBreaker.pointerId);
            connectionBreaker = null;
        }
    }

    function cancelSelection() {
        if (selection) {
            mainContainerEl.releasePointerCapture(selection.pointerId);
            selection = null;
        }
    }

    function onPointerDown(event: PointerEvent) {
        if (event.button === 1) {
            panning = true;
            panStartX = event.clientX;
            panStartY = event.clientY;
            startViewX = graphView.viewPosX;
            startViewY = graphView.viewPosY;
            mainContainerEl.setPointerCapture(event.pointerId);
        } else if (
            event.button === 2 &&
            event.buttons === 2 &&
            event.target === mainContainerEl
        ) {
            graphView.selectedNodeIds = new Set();
            updateSelection();
            connectionBreaker = {
                points: [{x: event.clientX, y: event.clientY}],
                pointerId: event.pointerId
            };
            mainContainerEl.setPointerCapture(event.pointerId);
            event.preventDefault();
        } else if (connectionBreaker && event.button !== 2) {
            cancelBreaker();
            event.preventDefault();
        } else if (event.button === 0 && event.target === mainContainerEl) {
            selection = {
                startX: event.clientX,
                startY: event.clientY,
                x: event.clientX,
                y: event.clientY,
                pointerId: event.pointerId,
                rect: mainContainerEl.getBoundingClientRect()
            };
            mainContainerEl.setPointerCapture(event.pointerId);
        } else if (selection && event.button !== 0) {
            cancelSelection();
        }
    }

    function onPointerMove(event: PointerEvent) {
        if (panning) {
            graphView.viewPosX = startViewX + (event.clientX - panStartX);
            graphView.viewPosY = startViewY + (event.clientY - panStartY);
            invoke('update_graph', {
                viewScale: graphView.viewScale,
                viewPosX: graphView.viewPosX,
                viewPosY: graphView.viewPosY
            });
        } else if (connectionBreaker) {
            if (event.buttons === 2) {
                connectionBreaker = {
                    ...connectionBreaker,
                    points: [
                        ...connectionBreaker.points,
                        {x: event.clientX, y: event.clientY}
                    ]
                };
            } else {
                cancelBreaker();
            }
        } else if (selection && event.pointerId === selection.pointerId) {
            selection = {...selection, x: event.clientX, y: event.clientY};
        } else if (newNodeDrag && event.pointerId === newNodeDrag.pointerId) {
            const rect = mainContainerEl.getBoundingClientRect();
            const nx = (event.clientX - rect.left - graphView.viewPosX) / graphView.viewScale;
            const ny = (event.clientY - rect.top - graphView.viewPosY) / graphView.viewScale;
            const node = graphView.nodes.find((n) => n.id === newNodeDrag?.nodeId);
            if (node) {
                dragNode({nodeId: node.id, dx: nx - node.viewPosX, dy: ny - node.viewPosY});
            }
        }
    }

    function onPointerUp(event: PointerEvent) {
        if (event.button === 1 && panning) {
            panning = false;
            mainContainerEl.releasePointerCapture(event.pointerId);
            invoke('update_graph', {
                viewScale: graphView.viewScale,
                viewPosX: graphView.viewPosX,
                viewPosY: graphView.viewPosY
            });
        } else if (event.button === 2 && connectionBreaker) {
            const rect = mainContainerEl.getBoundingClientRect();
            const pts = connectionBreaker.points.map((p) => ({
                x: (p.x - rect.left - graphView.viewPosX) / graphView.viewScale,
                y: (p.y - rect.top - graphView.viewPosY) / graphView.viewScale
            }));
            const before = [...graphView.connections];
            graphView.connections = graphView.connections.filter((c) => {
                const start = getOutputPinPos(c.fromNodeId, c.fromIndex);
                const end = getInputPinPos(c.toNodeId, c.toIndex);
                const bez = connectionSegments(start, end);
                for (let i = 0; i < pts.length - 1; i++) {
                    for (let j = 0; j < bez.length - 1; j++) {
                        if (segmentsIntersect(pts[i], pts[i + 1], bez[j], bez[j + 1])) {
                            return false;
                        }
                    }
                }
                return true;
            });
            const removed = before.filter(
                (b) =>
                    !graphView.connections.some(
                        (c) =>
                            c.fromNodeId === b.fromNodeId &&
                            c.fromIndex === b.fromIndex &&
                            c.toNodeId === b.toNodeId &&
                            c.toIndex === b.toIndex
                    )
            );
            if (removed.length > 0) {
                invoke('remove_connections_from_graph_view', {connections: removed})
                    .then(() => verifyGraphView())
                    .catch((e) => {
                        console.error('Failed to remove connections', e);
                    });
            }
            cancelBreaker();
        } else if (
            event.button === 0 &&
            selection &&
            event.pointerId === selection.pointerId
        ) {
            const {startX, startY, x, y, rect} = selection;
            const sx = Math.min(startX, x) - rect.left;
            const sy = Math.min(startY, y) - rect.top;
            const ex = Math.max(startX, x) - rect.left;
            const ey = Math.max(startY, y) - rect.top;

            const nodesEls = Array.from(
                mainContainerEl.querySelectorAll('[data-node-id]')
            ) as HTMLElement[];
            const ids: string[] = [];
            for (const el of nodesEls) {
                const r = el.getBoundingClientRect();
                const left = r.left - rect.left;
                const top = r.top - rect.top;
                const right = r.right - rect.left;
                const bottom = r.bottom - rect.top;
                if (left <= ex && right >= sx && top <= ey && bottom >= sy) {
                    const idAttr = el.getAttribute('data-node-id');
                    if (idAttr) ids.push(String(idAttr));
                }
            }
            graphView.selectedNodeIds = new Set(ids);
            updateSelection();
            cancelSelection();
        } else if (selection && event.button !== 0) {
            cancelSelection();
        } else if (newNodeDrag && event.pointerId === newNodeDrag.pointerId) {
            endDragNode(newNodeDrag.nodeId);
            newNodeDrag = null;
            mainContainerEl.releasePointerCapture(event.pointerId);
        }
    }

    function resetZoom() {
        const rect = mainContainerEl.getBoundingClientRect();
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        const gx = (cx - graphView.viewPosX) / graphView.viewScale;
        const gy = (cy - graphView.viewPosY) / graphView.viewScale;
        const newScale = 1;

        graphView.viewScale = newScale;
        graphView.viewPosX = cx - gx * newScale;
        graphView.viewPosY = cy - gy * newScale;
        invoke('update_graph', {
            viewScale: graphView.viewScale,
            viewPosX: graphView.viewPosX,
            viewPosY: graphView.viewPosY
        });
    }

    function centerView() {
        const nodes = Array.from(
            mainContainerEl.querySelectorAll('[data-node-id]')
        ) as HTMLElement[];
        if (nodes.length === 0) return;
        const containerRect = mainContainerEl.getBoundingClientRect();
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const el of nodes) {
            const r = el.getBoundingClientRect();
            const left = (r.left - containerRect.left - graphView.viewPosX) / graphView.viewScale;
            const top = (r.top - containerRect.top - graphView.viewPosY) / graphView.viewScale;
            const right = left + r.width / graphView.viewScale;
            const bottom = top + r.height / graphView.viewScale;
            if (left < minX) minX = left;
            if (top < minY) minY = top;
            if (right > maxX) maxX = right;
            if (bottom > maxY) maxY = bottom;
        }
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        graphView.viewPosX = containerRect.width / 2 - centerX * graphView.viewScale;
        graphView.viewPosY = containerRect.height / 2 - centerY * graphView.viewScale;
        invoke('update_graph', {
            viewScale: graphView.viewScale,
            viewPosX: graphView.viewPosX,
            viewPosY: graphView.viewPosY
        });
    }

    function fitView() {
        const nodes = Array.from(
            mainContainerEl.querySelectorAll('[data-node-id]')
        ) as HTMLElement[];
        if (nodes.length === 0) return;
        const containerRect = mainContainerEl.getBoundingClientRect();
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        for (const el of nodes) {
            const r = el.getBoundingClientRect();
            const left = (r.left - containerRect.left - graphView.viewPosX) / graphView.viewScale;
            const top = (r.top - containerRect.top - graphView.viewPosY) / graphView.viewScale;
            const right = left + r.width / graphView.viewScale;
            const bottom = top + r.height / graphView.viewScale;
            if (left < minX) minX = left;
            if (top < minY) minY = top;
            if (right > maxX) maxX = right;
            if (bottom > maxY) maxY = bottom;
        }
        const width = maxX - minX;
        const height = maxY - minY;
        if (width === 0 || height === 0) return;
        const margin = 40;
        const scaleX = (containerRect.width - margin) / width;
        const scaleY = (containerRect.height - margin) / height;
        const newScale = Math.min(Math.max(Math.min(scaleX, scaleY), 0.2), 4);
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        graphView.viewScale = newScale;
        graphView.viewPosX = containerRect.width / 2 - centerX * newScale;
        graphView.viewPosY = containerRect.height / 2 - centerY * newScale;
        invoke('update_graph', {
            viewScale: graphView.viewScale,
            viewPosX: graphView.viewPosX,
            viewPosY: graphView.viewPosY
        });
    }
</script>

<div class=" flex flex-col h-full overflow-hidden">

    <div
            class="flex-1 relative overflow-hidden graph-bg bg-base-100"
            style="background-position: {bgX()}px {bgY()}px; background-size: {dotSpacing()}px {dotSpacing()}px;"
            id="main-container"
            bind:this={mainContainerEl}
            onwheel={onWheel}
            onpointerdown={onPointerDown}
            onpointermove={onPointerMove}
            onpointerup={onPointerUp}
            onpointerleave={onPointerUp}
            oncontextmenu={onContextMenu}
            role="region"
    >
        <svg
                class="absolute top-0 left-0 w-full h-full pointer-events-none text-primary overflow-visible"
                style="transform: translate({graphView.viewPosX}px, {graphView.viewPosY}px) scale({graphView.viewScale}); transform-origin: 0 0;"
        >
            {#each connectionPaths as _c, i}
                <path d={connectionPaths[i]} stroke="currentColor" fill="none" stroke-width="2"/>
            {/each}
            {#if pendingConnection}
                <path d={pendingConnectionPath} stroke="currentColor" fill="none" stroke-width="2"/>
            {/if}

        </svg>
        <svg class="absolute top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
            {#if connectionBreaker}
                <path d={breakerPath()} stroke="red" fill="none" stroke-width="2"/>
            {/if}
        </svg>
        {#if selectionBox()}
            <div id="selection-box"
                 class="absolute border-2 border-blue-500 bg-blue-500/25 pointer-events-none"
                 style="left: {selectionBox()?.left}px; top: {selectionBox()?.top}px; width: {selectionBox()?.width}px; height: {selectionBox()?.height}px;"
            ></div>
        {/if}
        <div style="transform: translate({graphView.viewPosX}px, {graphView.viewPosY}px) scale({graphView.viewScale}); transform-origin: 0 0;">
            {#each graphView.nodes as node, i}
                <Node
                        nodeView={node}
                        connectionStart={startConnection}
                        connectionEnd={endConnection}
                        drag={dragNode}
                        dragEnd={endDragNode}
                        viewScale={graphView.viewScale}
                        viewPosX={graphView.viewPosX}
                        viewPosY={graphView.viewPosY}
                        selected={graphView.selectedNodeIds.has(node.id)}
                        select={onNodeSelect}
                        remove={removeNode}
                />
            {/each}
        </div>
        <button class="btn btn-xs absolute top-2 left-2 w-5 h-5" onclick={() => showFuncLibrary = true}>+</button>
    </div>

    {#if showFuncLibrary}
        <FuncLibrary close={() => (showFuncLibrary = false)} startDrag={startFuncDrag} container={mainContainerEl}/>
    {/if}

    <div class="h-8 border-base-300 bg-base-100 text-xs flex items-center px-2 py-1">
        <button class="btn btn-xs p-1 w-8" onclick={resetZoom}>{zoomPercent()}%</button>
        <button class="btn btn-xs p-1" onclick={centerView}>Center</button>
        <button class="btn btn-xs p-1" onclick={fitView}>Fit</button>
    </div>
</div>

<style>
    .graph-bg {
        background-image: radial-gradient(circle, rgba(255, 255, 255, 0.075) 1px, transparent 1px);
    }
</style>