//! The document currently open in a frontend, including its persistence path
//! and whether derived graph state needs normalization.

use std::path::{Path, PathBuf};

use scenarium::Library;

use crate::core::document::Document;
use crate::core::io::document::{self, DocumentLoadError, DocumentSaveError};

#[derive(Debug)]
pub(crate) struct OpenDocument {
    pub(crate) document: Document,
    pub(crate) path: Option<PathBuf>,
    pub(crate) normalization_pending: bool,
}

impl OpenDocument {
    pub(crate) fn load(path: PathBuf) -> Result<Self, DocumentLoadError> {
        let document = document::load(&path)?;
        Ok(Self {
            document,
            path: Some(path),
            normalization_pending: true,
        })
    }

    pub(crate) fn normalize(&mut self, library: &Library) {
        if !self.normalization_pending {
            return;
        }
        self.document.normalize(library);
        self.normalization_pending = false;
    }

    pub(crate) fn save_to(
        &mut self,
        path: &Path,
        library: &Library,
    ) -> Result<(), DocumentSaveError> {
        self.normalize(library);
        document::save(&self.document, path)?;
        self.path = Some(path.to_path_buf());
        Ok(())
    }
}

impl Default for OpenDocument {
    fn default() -> Self {
        Self {
            document: Document::default(),
            path: None,
            normalization_pending: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use glam::Vec2;
    use scenarium::testing;
    use scenarium::{Binding, Func, FuncId, FuncInput, InputPort, Library, StaticValue};
    use scenarium::{DataType, FuncOutput, Graph, GraphId, GraphLink, Node, NodeKind, OutputPort};

    use crate::core::document::open_document::OpenDocument;
    use crate::core::document::{Document, ItemRef, PortKind, PortRef, TabRef};
    use crate::core::io::document::DocumentLoadError;

    fn from_document(document: Document) -> OpenDocument {
        OpenDocument {
            document,
            path: None,
            normalization_pending: true,
        }
    }

    #[test]
    fn load_returns_the_document_error() {
        let path = PathBuf::from("not-a-document.json");

        let error = OpenDocument::load(path.clone()).unwrap_err();

        assert!(matches!(
            error,
            DocumentLoadError::InvalidExtension { path: error_path } if error_path == path
        ));
    }

    #[test]
    fn normalization_prunes_stale_wiring_once_for_each_open_document() {
        let func_id = FuncId::unique();
        let previous = Func::new(func_id, "changed")
            .input(FuncInput::required("removed", scenarium::DataType::Float));
        let library = Library::from([testing::with_stub_lambda(Func::new(func_id, "changed"))]);
        let document = || {
            let mut graph = Graph::default();
            let node_id = graph.add_func_node(&previous);
            let input = InputPort::new(node_id, 0);
            graph.set_input_binding(input, Binding::Const(StaticValue::Float(3.0)));
            Document::from(graph)
        };
        let mut open = from_document(document());

        assert!(open.normalization_pending);
        assert_eq!(open.document.graph.bindings.len(), 1);
        open.normalize(&library);
        assert!(!open.normalization_pending);
        assert!(open.document.graph.bindings.is_empty());

        open = from_document(document());
        assert!(open.normalization_pending);
        assert_eq!(open.document.graph.bindings.len(), 1);
        open.normalize(&library);
        assert!(!open.normalization_pending);
        assert!(open.document.graph.bindings.is_empty());
    }

    #[test]
    fn normalization_remaps_pinned_views_and_viewer_tabs() {
        let producer = testing::with_stub_lambda(
            Func::new(FuncId::unique(), "producer").output(FuncOutput::new("value", DataType::Int)),
        );
        let library = Library::from([producer.clone()]);
        let mut nested = Graph::new("nested").outputs([
            FuncOutput::new("A", DataType::Int),
            FuncOutput::new("B", DataType::Int),
            FuncOutput::new("C", DataType::Int),
        ]);
        let producer_id = nested.add_func_node(&producer);
        let output_id = nested.add(Node::new(NodeKind::GraphOutput));
        nested.set_input_binding(InputPort::new(output_id, 0), Binding::bind(producer_id, 0));
        nested.set_input_binding(InputPort::new(output_id, 2), Binding::bind(producer_id, 0));

        let graph_id = GraphId::unique();
        let mut graph = Graph::default();
        graph.insert_graph(graph_id, nested.clone());
        let instance = graph.add_graph_node(&nested, GraphLink::Local(graph_id));
        let removed = OutputPort::new(instance, 1);
        let remapped = OutputPort::new(instance, 2);
        let normalized = OutputPort::new(instance, 1);
        graph.set_output_pinned(removed, true);
        graph.set_output_pinned(remapped, true);

        let mut document = Document::from(graph);
        document
            .main_view
            .item_placements
            .insert(ItemRef::Pin(removed), Vec2::new(10.0, 11.0));
        document
            .main_view
            .item_placements
            .insert(ItemRef::Pin(remapped), Vec2::new(20.0, 21.0));
        document.main_view.selected = [ItemRef::Pin(removed), ItemRef::Pin(remapped)]
            .into_iter()
            .collect();
        let primary = document.layout.primary().id;
        let viewer = |port: OutputPort| {
            TabRef::ImageViewer(PortRef {
                node_id: port.node_id,
                kind: PortKind::Output,
                port_idx: port.port_idx,
            })
        };
        document.layout.find_or_insert(viewer(removed), primary);
        document.layout.find_or_insert(viewer(remapped), primary);
        document.layout.activate(primary, 2);
        let mut open = from_document(document);

        open.normalize(&library);

        assert_eq!(
            open.document.graph.pinned_outputs().collect::<Vec<_>>(),
            [normalized]
        );
        assert_eq!(
            open.document
                .main_view
                .item_placements
                .get(&ItemRef::Pin(normalized)),
            Some(&Vec2::new(20.0, 21.0))
        );
        assert_eq!(
            open.document.main_view.selected,
            [ItemRef::Pin(normalized)].into_iter().collect()
        );
        assert_eq!(
            open.document
                .layout
                .all_tabs()
                .filter(|tab| matches!(tab, TabRef::ImageViewer(_)))
                .collect::<Vec<_>>(),
            [viewer(normalized)]
        );
        assert_eq!(
            open.document.layout.primary().active_tab(),
            viewer(normalized)
        );
        open.document.validate().unwrap();
    }

    #[test]
    fn empty_document_has_the_main_graph_tab() {
        let open = OpenDocument::default();

        assert!(open.path.is_none());
        assert_eq!(open.document.layout.all_tabs().count(), 1);
        assert!(open.normalization_pending);
    }
}
