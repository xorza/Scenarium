use std::mem::ManuallyDrop;

use graph::function::FuncId;

use crate::{get_context, FfiBuf, FfiId, FfiStr};

#[repr(C)]
struct FfiInput {
    a: u8,
}

#[repr(C)]
struct FfiNode {
    id: FfiId,
    func_id: FfiId,
    name: FfiStr,
    is_output: bool,
    cache_outputs: bool,
    inputs: FfiBuf,
    events: FfiBuf,
}

impl From<&graph::graph::Node> for FfiNode {
    fn from(node: &graph::graph::Node) -> Self {
        FfiNode {
            id: node.id.as_uuid().into(),
            func_id: node.func_id.as_uuid().into(),
            name: node.name.clone().into(),
            is_output: node.is_output,
            cache_outputs: node.cache_outputs,
            inputs: FfiBuf::default(),
            events: FfiBuf::default(),
        }
    }
}

#[no_mangle]
extern "C" fn get_nodes(ctx: *mut u8) -> FfiBuf {
    // let graph = Graph::from_yaml(include_str!("../../test_resources/test_graph.yml")).unwrap();

    get_context(ctx)
        .graph
        .nodes()
        .iter()
        .map(FfiNode::from)
        .collect::<Vec<FfiNode>>()
        .into()
}

#[no_mangle]
extern "C" fn new_node(ctx: *mut u8, func_id: FfiId) -> FfiNode {
    let context = get_context(ctx);
    let func_id: FuncId = ManuallyDrop::new(func_id).to_uuid().into();

    let func = context.func_lib.func_by_id(func_id).unwrap();
    let node = graph::graph::Node::from_function(func);
    context.graph.add_node(node);

    context.graph.nodes().last().unwrap().into()
}

#[no_mangle]
extern "C" fn dummy1(_a: FfiNode, _b: FfiInput) {}