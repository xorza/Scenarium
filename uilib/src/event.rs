use glam::UVec2;

#[derive(PartialEq, Debug, Clone)]
pub enum Event {
    Resize {
        size: UVec2,
    },
    WindowClose,
    Unknown,
}

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum EventResult {
    Continue,
    Exit,
}

impl<'a, T> From<winit::event::Event<'a, T>> for Event {
    fn from(event: winit::event::Event<'a, T>) -> Self {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(size) => Event::Resize {
                    size: UVec2::new(size.width.max(1), size.height.max(1)),
                },
                winit::event::WindowEvent::Focused(_is_focused) => {
                    Event::Unknown
                }
                winit::event::WindowEvent::CursorEntered { .. } => {
                    Event::Unknown
                }
                winit::event::WindowEvent::CursorLeft { .. } => {
                    Event::Unknown
                }
                winit::event::WindowEvent::CursorMoved { position: _position, .. } => {
                    Event::Unknown
                }
                winit::event::WindowEvent::Occluded(_is_occluded) => {
                    Event::Unknown
                }
                winit::event::WindowEvent::MouseInput { state: _state, button: _button, .. } => {
                    Event::Unknown
                }
                winit::event::WindowEvent::MouseWheel { delta: _delta, phase: _phase, .. } => {
                    Event::Unknown
                }
                winit::event::WindowEvent::CloseRequested => {
                    Event::WindowClose
                }
                _ => Event::Unknown,
            },
            _ => Event::Unknown,
        }
    }
}