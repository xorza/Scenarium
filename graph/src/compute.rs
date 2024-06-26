use std::fmt::Debug;
use std::ops::{Index, IndexMut};

use common::apply::ApplyMut;

use crate::data::{DataType, DynamicValue};
use crate::function::FuncLib;
use crate::graph::{Binding, Graph};
use crate::invoke::Invoker;
use crate::runtime_graph::RuntimeGraph;

#[derive(Default)]
pub(crate) struct ArgSet(Vec<DynamicValue>);

#[derive(Debug, Default)]
pub struct Compute {}

impl Compute {
    pub fn run<T>(
        &self,
        graph: &Graph,
        func_lib: &FuncLib,
        invoker: &T,
        runtime_graph: &mut RuntimeGraph,
    ) -> anyhow::Result<()>
    where
        T: Invoker,
    {
        runtime_graph.next(graph);

        let mut inputs: ArgSet = ArgSet::default();

        let active_node_indexes = runtime_graph
            .nodes
            .iter_mut()
            .enumerate()
            .filter_map(|(index, r_node)| {
                if r_node.should_invoke {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<usize>>();

        for node_idx in active_node_indexes {
            let node = graph
                .node_by_id(runtime_graph.nodes[node_idx].id())
                .unwrap();
            let node_info = func_lib.func_by_id(node.func_id).unwrap();

            inputs.resize_and_fill(node.inputs.len());
            node.inputs
                .iter()
                .enumerate()
                .map(|(input_idx, input)| {
                    let value: DynamicValue = match &input.binding {
                        Binding::None => DynamicValue::None,
                        Binding::Const => input
                            .const_value
                            .as_ref()
                            .expect("Const value is not set")
                            .into(),

                        Binding::Output(output_binding) => {
                            let output_r_node = runtime_graph
                                .node_by_id_mut(output_binding.output_node_id)
                                .unwrap();

                            output_r_node
                                .decrement_current_binding_count(output_binding.output_index);

                            let output_values = output_r_node.output_values.as_mut().unwrap();
                            let value = output_values
                                .get_mut(output_binding.output_index as usize)
                                .unwrap();

                            value.clone()
                        }
                    };

                    let data_type = &node_info.inputs[input_idx].data_type;

                    (input_idx, data_type, value)
                })
                .for_each(|(input_idx, data_type, value)| {
                    inputs[input_idx] = self.convert_type(&value, data_type);
                });

            let r_node = &mut runtime_graph.nodes[node_idx];
            let outputs = r_node
                .output_values
                .get_or_insert_with(|| vec![DynamicValue::None; node_info.outputs.len()]);

            r_node.run_time = {
                let start = std::time::Instant::now();
                invoker.invoke(
                    node.func_id,
                    &mut r_node.cache,
                    inputs.as_mut_slice(),
                    outputs.as_mut_slice(),
                )?;

                start.elapsed().as_secs_f64()
            };

            inputs.fill();
        }

        for r_node in runtime_graph.nodes.iter_mut() {
            if !r_node.should_cache_outputs {
                r_node
                    .output_values
                    .as_mut()
                    .apply_mut(|values| values.fill(DynamicValue::None));
            }

            assert_eq!(r_node.total_binding_count, 0);
        }

        Ok(())
    }

    fn convert_type(&self, src_value: &DynamicValue, dst_data_type: &DataType) -> DynamicValue {
        let src_data_type = src_value.data_type();
        if *src_data_type == *dst_data_type {
            return src_value.clone();
        }

        if src_data_type.is_custom() || dst_data_type.is_custom() {
            panic!("Custom types are not supported yet");
        }

        match (src_data_type, dst_data_type) {
            (DataType::Bool, DataType::Int) => DynamicValue::Int(src_value.as_bool() as i64),
            (DataType::Bool, DataType::Float) => {
                DynamicValue::Float(src_value.as_bool() as i64 as f64)
            }
            (DataType::Bool, DataType::String) => {
                DynamicValue::String(src_value.as_bool().to_string())
            }

            (DataType::Int, DataType::Bool) => DynamicValue::Bool(src_value.as_int() != 0),
            (DataType::Int, DataType::Float) => DynamicValue::Float(src_value.as_int() as f64),
            (DataType::Int, DataType::String) => {
                DynamicValue::String(src_value.as_int().to_string())
            }

            (DataType::Float, DataType::Bool) => {
                DynamicValue::Bool(src_value.as_float().abs() > common::EPSILON)
            }
            (DataType::Float, DataType::Int) => DynamicValue::Int(src_value.as_float() as i64),
            (DataType::Float, DataType::String) => {
                DynamicValue::String(src_value.as_float().to_string())
            }

            (src, dst) => {
                panic!("Unsupported conversion from {:?} to {:?}", src, dst);
            }
        }
    }
}

impl ArgSet {
    pub(crate) fn from_vec<T>(vec: Vec<T>) -> Self
    where
        T: Into<DynamicValue>,
    {
        ArgSet(vec.into_iter().map(|v| v.into()).collect())
    }
    pub(crate) fn resize_and_fill(&mut self, size: usize) {
        self.0.resize(size, DynamicValue::None);
        self.fill();
    }
    pub(crate) fn fill(&mut self) {
        self.0.fill(DynamicValue::None);
    }
    pub(crate) fn as_slice(&self) -> &[DynamicValue] {
        self.0.as_slice()
    }
    pub(crate) fn as_mut_slice(&mut self) -> &mut [DynamicValue] {
        self.0.as_mut_slice()
    }
}
impl Index<usize> for ArgSet {
    type Output = DynamicValue;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for ArgSet {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
