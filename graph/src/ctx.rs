use crate::compute::Compute;
use crate::function::FuncLib;
use crate::graph::Graph;
use crate::invoke::UberInvoker;
use crate::runtime_graph::RuntimeGraph;
use crate::worker::Worker;

#[derive(Default, Debug)]
pub struct Context {
    pub(crate) graph: Graph,
    pub(crate) runtime_graph: RuntimeGraph,
    pub(crate) compute: Compute,
    pub(crate) func_lib: FuncLib,

    pub(crate) invoker: Option<UberInvoker>,
    pub(crate) worker: Option<Worker>,
}
