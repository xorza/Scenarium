//! Keep graph interfaces and wiring consistent with the graph tree and library.

use std::collections::HashMap;

use crate::data::type_system::DataType;
use crate::graph::interface::{GraphEvent, GraphId, GraphLink};
use crate::graph::{
    Binding, Graph, InputPort, NodeId, NodeKind, NodeSearch, OutputPort, Subscription,
};
use crate::library::Library;
use crate::node::definition::{FuncInput, FuncOutput, OutputType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputPortChange {
    pub from: OutputPort,
    pub to: Option<OutputPort>,
}

/// Output-port identity changes made while normalizing a graph.
///
/// Graph-owned references are updated internally. Callers that keep state keyed
/// by [`OutputPort`] outside the graph must apply these changes: `Some` remaps
/// the key and `None` removes it.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct NormalizationReport {
    pub output_ports: Vec<OutputPortChange>,
}

#[derive(Debug)]
struct SidePlan<T> {
    boundary: NodeId,
    interface: Vec<T>,
    remap: HashMap<usize, usize>,
    old_len: usize,
    changed: bool,
}

#[derive(Debug)]
struct EventPlan {
    interface: Vec<GraphEvent>,
    remap: HashMap<usize, usize>,
    changed: bool,
}

impl Graph {
    /// Restore derived graph interfaces and remove wiring that no longer
    /// resolves against the resulting graph tree and library, returning
    /// identity changes for caller-owned state.
    pub fn normalize(&mut self, library: &Library) -> NormalizationReport {
        let mut report = NormalizationReport::default();
        self.normalize_interfaces(library, &mut report);
        self.prune_dangling_references(library, &mut report);
        report
    }

    fn normalize_interfaces(&mut self, library: &Library, report: &mut NormalizationReport) {
        let graph_ids: Vec<GraphId> = self.graphs.keys().copied().collect();
        for graph_id in graph_ids {
            self.graphs
                .get_mut(&graph_id)
                .unwrap()
                .normalize_interfaces(library, report);
            normalize_local_graph(self, graph_id, library, report);
        }
    }

    fn prune_dangling_references(&mut self, library: &Library, report: &mut NormalizationReport) {
        for graph in self.graphs.values_mut() {
            graph.prune_dangling_references(library, report);
        }

        let mut bindings = std::mem::take(&mut self.bindings);
        bindings.retain(|destination, binding| self.binding_live(*destination, binding, library));
        self.bindings = bindings;

        let mut subscriptions = std::mem::take(&mut self.subscriptions);
        subscriptions.retain(|subscription| self.subscription_live(subscription, library));
        self.subscriptions = subscriptions;

        let mut pinned_outputs = std::mem::take(&mut self.pinned_outputs);
        pinned_outputs.retain(|port| {
            let live = self.output_live(*port, library);
            if !live {
                report.output_ports.push(OutputPortChange {
                    from: *port,
                    to: None,
                });
            }
            live
        });
        self.pinned_outputs = pinned_outputs;

        if self.definition.is_some() {
            let mut events = std::mem::take(&mut self.definition_mut().events);
            events.retain(|event| self.event_live(event.emitter, event.emitter_event_idx, library));
            self.definition_mut().events = events;
        }
    }

    fn input_live(&self, port: InputPort, library: &Library) -> bool {
        self.find(&port.node_id, NodeSearch::TopLevel)
            .is_some_and(|node| {
                self.input_count(node, library)
                    .is_none_or(|count| port.port_idx < count)
            })
    }

    fn output_live(&self, port: OutputPort, library: &Library) -> bool {
        self.find(&port.node_id, NodeSearch::TopLevel)
            .is_some_and(|node| {
                self.output_count(node, library)
                    .is_none_or(|count| port.port_idx < count)
            })
    }

    fn event_live(&self, emitter: NodeId, event_idx: usize, library: &Library) -> bool {
        self.find(&emitter, NodeSearch::TopLevel)
            .is_some_and(|node| {
                self.event_count(node, library)
                    .is_none_or(|count| event_idx < count)
            })
    }

    fn binding_live(&self, destination: InputPort, binding: &Binding, library: &Library) -> bool {
        self.input_live(destination, library)
            && match binding {
                Binding::Bind(source) => self.output_live(*source, library),
                Binding::Const(_) => true,
            }
    }

    fn subscription_live(&self, subscription: &Subscription, library: &Library) -> bool {
        self.event_live(subscription.emitter, subscription.event_idx, library)
            && self
                .find(&subscription.subscriber, NodeSearch::TopLevel)
                .is_some()
    }
}

fn normalize_local_graph(
    parent: &mut Graph,
    graph_id: GraphId,
    library: &Library,
    report: &mut NormalizationReport,
) {
    let (inputs, outputs, events) = {
        let graph = parent.graphs.get(&graph_id).unwrap();
        (
            plan_inputs(graph, library),
            plan_outputs(graph, library),
            plan_events(graph, library),
        )
    };
    let input_changed = inputs.as_ref().is_some_and(|plan| plan.changed);
    let output_changed = outputs.as_ref().is_some_and(|plan| plan.changed);
    if !input_changed && !output_changed && !events.changed {
        return;
    }

    let instances: Vec<NodeId> = parent
        .iter()
        .filter_map(|node| match node.kind {
            NodeKind::Graph(GraphLink::Local(id)) if id == graph_id => Some(node.id),
            _ => None,
        })
        .collect();

    let graph = parent.graphs.get_mut(&graph_id).unwrap();
    if let Some(plan) = &inputs {
        remap_output_references(graph, plan.boundary, plan, report);
        graph.definition_mut().inputs = plan.interface.clone();
    }
    if let Some(plan) = &outputs {
        remap_target_bindings(graph, plan.boundary, &plan.remap);
        graph.definition_mut().outputs = plan.interface.clone();
    }
    if events.changed {
        graph.definition_mut().events = events.interface.clone();
    }

    for node_id in instances {
        if let Some(plan) = &inputs {
            remap_target_bindings(parent, node_id, &plan.remap);
        }
        if let Some(plan) = &outputs {
            remap_output_references(parent, node_id, plan, report);
        }
        if events.changed {
            remap_event_references(parent, node_id, &events.remap);
        }
    }
}

fn plan_inputs(graph: &Graph, library: &Library) -> Option<SidePlan<FuncInput>> {
    let boundary = graph
        .iter()
        .find(|node| matches!(node.kind, NodeKind::GraphInput))
        .map(|node| node.id)?;
    let used = used_sorted(
        graph
            .edges()
            .filter_map(|(_, source)| (source.node_id == boundary).then_some(source.port_idx)),
    );
    let mut remap = HashMap::with_capacity(used.len());
    let mut interface = Vec::with_capacity(used.len());
    for (new_idx, &old_idx) in used.iter().enumerate() {
        remap.insert(old_idx, new_idx);
        let data_type = infer_used_input_type(graph, library, boundary, old_idx);
        interface.push(match graph.definition().inputs.get(old_idx) {
            Some(existing) => FuncInput {
                data_type,
                ..existing.clone()
            },
            None => synth_input(new_idx, data_type),
        });
    }
    let changed = interface != graph.definition().inputs
        || remap.iter().any(|(old, new)| old != new);
    Some(SidePlan {
        boundary,
        interface,
        remap,
        old_len: graph.definition().inputs.len(),
        changed,
    })
}

fn plan_outputs(graph: &Graph, library: &Library) -> Option<SidePlan<FuncOutput>> {
    let boundary = graph
        .iter()
        .find(|node| matches!(node.kind, NodeKind::GraphOutput))
        .map(|node| node.id)?;
    let used = used_sorted(
        graph
            .bindings_touching(boundary)
            .into_iter()
            .filter_map(|entry| (entry.port.node_id == boundary).then_some(entry.port.port_idx)),
    );
    let mut remap = HashMap::with_capacity(used.len());
    let mut interface = Vec::with_capacity(used.len());
    for (new_idx, &old_idx) in used.iter().enumerate() {
        remap.insert(old_idx, new_idx);
        let data_type = infer_used_output_type(graph, library, boundary, old_idx);
        interface.push(match graph.definition().outputs.get(old_idx) {
            Some(existing) => FuncOutput {
                ty: OutputType::Fixed(data_type),
                ..existing.clone()
            },
            None => FuncOutput::new(format!("output{new_idx}"), data_type),
        });
    }
    let changed = interface != graph.definition().outputs
        || remap.iter().any(|(old, new)| old != new);
    Some(SidePlan {
        boundary,
        interface,
        remap,
        old_len: graph.definition().outputs.len(),
        changed,
    })
}

fn plan_events(graph: &Graph, library: &Library) -> EventPlan {
    let mut interface = Vec::with_capacity(graph.definition().events.len());
    let mut remap = HashMap::with_capacity(graph.definition().events.len());
    for (old_idx, event) in graph.definition().events.iter().enumerate() {
        if graph.event_live(event.emitter, event.emitter_event_idx, library) {
            remap.insert(old_idx, interface.len());
            interface.push(event.clone());
        }
    }
    let changed = interface != graph.definition().events
        || remap.iter().any(|(old_idx, new_idx)| old_idx != new_idx);
    EventPlan {
        interface,
        remap,
        changed,
    }
}

fn remap_output_references<T>(
    graph: &mut Graph,
    node_id: NodeId,
    plan: &SidePlan<T>,
    report: &mut NormalizationReport,
) {
    for old_idx in 0..plan.old_len {
        let new_idx = plan.remap.get(&old_idx).copied();
        if new_idx != Some(old_idx) {
            report.output_ports.push(OutputPortChange {
                from: OutputPort::new(node_id, old_idx),
                to: new_idx.map(|idx| OutputPort::new(node_id, idx)),
            });
        }
    }

    let edges: Vec<(InputPort, usize)> = graph
        .edges()
        .filter(|(_, source)| source.node_id == node_id)
        .map(|(target, source)| (target, source.port_idx))
        .collect();
    for (target, old_idx) in edges {
        match plan.remap.get(&old_idx) {
            Some(&new_idx) if new_idx != old_idx => {
                graph.set_input_binding(target, Binding::bind(node_id, new_idx));
            }
            Some(_) => {}
            None => graph.set_input_binding(target, None),
        }
    }

    graph.pinned_outputs = std::mem::take(&mut graph.pinned_outputs)
        .into_iter()
        .filter_map(|port| {
            if port.node_id != node_id {
                return Some(port);
            }
            plan.remap
                .get(&port.port_idx)
                .map(|&new_idx| OutputPort::new(node_id, new_idx))
        })
        .collect();
}

fn remap_event_references(graph: &mut Graph, node_id: NodeId, remap: &HashMap<usize, usize>) {
    let subscriptions = std::mem::take(&mut graph.subscriptions);
    graph.subscriptions = subscriptions
        .into_iter()
        .filter_map(|mut subscription| {
            if subscription.emitter != node_id {
                return Some(subscription);
            }
            subscription.event_idx = *remap.get(&subscription.event_idx)?;
            Some(subscription)
        })
        .collect();

    let events = std::mem::take(&mut graph.definition_mut().events);
    graph.definition_mut().events = events
        .into_iter()
        .filter_map(|mut event| {
            if event.emitter != node_id {
                return Some(event);
            }
            event.emitter_event_idx = *remap.get(&event.emitter_event_idx)?;
            Some(event)
        })
        .collect();
}

fn remap_target_bindings(graph: &mut Graph, node_id: NodeId, remap: &HashMap<usize, usize>) {
    let bindings: Vec<(usize, Binding)> = graph
        .bindings_touching(node_id)
        .into_iter()
        .filter(|entry| entry.port.node_id == node_id)
        .map(|entry| (entry.port.port_idx, entry.binding))
        .collect();
    if bindings.is_empty() {
        return;
    }
    for (old_idx, _) in &bindings {
        graph.set_input_binding(InputPort::new(node_id, *old_idx), None);
    }
    for (old_idx, binding) in bindings {
        if let Some(&new_idx) = remap.get(&old_idx) {
            graph.set_input_binding(InputPort::new(node_id, new_idx), binding);
        }
    }
}

fn used_sorted(indices: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut indices: Vec<usize> = indices.collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn synth_input(idx: usize, data_type: DataType) -> FuncInput {
    FuncInput::optional(format!("input{idx}"), data_type)
}

fn infer_used_input_type(
    graph: &Graph,
    library: &Library,
    boundary: NodeId,
    old_idx: usize,
) -> DataType {
    graph
        .edges()
        .find(|(_, source)| source.node_id == boundary && source.port_idx == old_idx)
        .and_then(|(target, _)| graph.input_type(library, target))
        .unwrap_or_default()
}

fn infer_used_output_type(
    graph: &Graph,
    library: &Library,
    boundary: NodeId,
    old_idx: usize,
) -> DataType {
    graph
        .bindings_touching(boundary)
        .into_iter()
        .find_map(|entry| match entry.binding {
            Binding::Bind(source)
                if entry.port.node_id == boundary && entry.port.port_idx == old_idx =>
            {
                Some(source)
            }
            _ => None,
        })
        .map(|source| graph.resolve_output_type(library, source))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
