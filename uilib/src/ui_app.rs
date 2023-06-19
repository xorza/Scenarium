use glam::UVec2;
use wgpu::{Adapter, Device, Queue, SurfaceConfiguration, TextureView};

use crate::app_base::App;
use crate::event::{Event, EventResult};

pub struct UiAppInternal {
    window_size: UVec2,
}

impl App for UiAppInternal {
    fn init(
        config: &SurfaceConfiguration,
        _adapter: &Adapter,
        _device: &Device,
        _queue: &Queue)
        -> Self {
        Self {
            window_size: UVec2::new(config.width, config.height),
        }
    }

    fn update(&mut self, event: Event) -> EventResult {
        match event {
            Event::WindowClose => EventResult::Exit,
            Event::Resize(size) => {
                self.window_size = size;

                EventResult::Redraw
            }

            _ => EventResult::Continue
        }
    }

    fn render(
        &mut self,
        _view: &TextureView,
        _device: &Device,
        _queue: &Queue,
        _time: f64) {
        todo!()
    }
}

impl UiAppInternal {
    pub fn window_size(&self) -> UVec2 {
        self.window_size
    }
}