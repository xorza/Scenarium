use std::ops::RangeBounds;

use bytemuck::{Pod, Zeroable};
use pollster::FutureExt;
use wgpu::util::DeviceExt;

use crate::image::{Image, ImageDesc};

pub(crate) struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub limits: wgpu::Limits,
    pub rect_one_vb: VertexBuffer,
}

fn aligned_size_of_uniform<U: Sized>() -> u64 {
    let uniform_size = std::mem::size_of::<U>();
    let uniform_align = 256;
    let uniform_padded_size = (uniform_size + uniform_align - 1) / uniform_align * uniform_align;

    uniform_padded_size as u64
}

impl WgpuContext {
    pub fn new() -> anyhow::Result<WgpuContext> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            dx12_shader_compiler: wgpu::Dx12Compiler::Dxc { dxil_path: None, dxc_path: None },
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .expect("Unable to find a suitable GPU adapter.");

        assert!(adapter.features().contains(wgpu::Features::PUSH_CONSTANTS));

        let _limits = adapter.limits();
        let limits = wgpu::Limits {
            max_push_constant_size: 256,
            max_texture_dimension_1d: 16384,
            max_texture_dimension_2d: 16384,
            ..Default::default()
        };

        let device_descriptor = wgpu::DeviceDescriptor {
            label: None,
            features: wgpu::Features::PUSH_CONSTANTS,
            limits: limits.clone(),
        };

        let (device, queue) = adapter
            .request_device(&device_descriptor, None)
            .block_on()
            .expect("Unable to find a suitable GPU device.");

        let rect_one_vb = VertexBuffer::from_slice(&device, &Vert2D::rect_one());

        Ok(WgpuContext {
            device,
            queue,
            limits,
            rect_one_vb,
        })
    }

    pub fn draw_one<T>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shader: &Shader,
        tex1_view: &wgpu::TextureView,
        tex2_view: &wgpu::TextureView,
        target_tex_view: &wgpu::TextureView,
        push_constant: &T,
    )
        where T: Pod
    {
        let device = &self.device;

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &shader.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(tex1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(tex2_view),
                }
            ],
            label: None,
        });

        {
            let mut render_pass = encoder.begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: target_tex_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        }),
                    ],
                    depth_stencil_attachment: None,
                    label: None,
                });

            render_pass.push_debug_group("Prepare data for draw.");

            render_pass.set_pipeline(&shader.pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_push_constants(
                wgpu::ShaderStages::VERTEX,
                0,
                bytemuck::bytes_of(push_constant),
            );

            render_pass.pop_debug_group();

            render_pass.insert_debug_marker("Draw.");
            render_pass.set_vertex_buffer(0, self.rect_one_vb.slice(..));
            render_pass.draw(0..self.rect_one_vb.vert_count, 0..1);


            drop(render_pass);
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct Vert2D(pub [f32; 2], pub [f32; 2]);

impl Vert2D {
    pub fn rect(width: f32, height: f32, tex_width: f32, tex_height: f32) -> [Vert2D; 4] {
        [
            Vert2D([0.0, 0.0], [0.0, 0.0]),
            Vert2D([width, 0.0], [tex_width, 0.0]),
            Vert2D([0.0, height], [0.0, tex_height]),
            Vert2D([width, height], [tex_width, tex_height]),
        ]
    }
    pub fn rect_one() -> [Vert2D; 4] {
        [
            Vert2D([0.0, 0.0], [0.0, 0.0]),
            Vert2D([1.0, 0.0], [1.0, 0.0]),
            Vert2D([0.0, 1.0], [0.0, 1.0]),
            Vert2D([1.0, 1.0], [1.0, 1.0]),
        ]
    }
}

pub(crate) struct VertexBuffer {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) vert_count: u32,
    pub(crate) stride: u32,
}

impl VertexBuffer {
    pub fn from_vec<V: Pod>(device: &wgpu::Device, data: Vec<V>) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX,
            label: None,
        });

        VertexBuffer {
            buffer,
            vert_count: data.len() as u32,
            stride: std::mem::size_of::<V>() as u32,
        }
    }
    pub fn from_slice<V: Pod>(device: &wgpu::Device, data: &[V]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::VERTEX,
            label: None,
        });

        VertexBuffer {
            buffer,
            vert_count: data.len() as u32,
            stride: std::mem::size_of::<V>() as u32,
        }
    }
    pub fn slice<S: RangeBounds<u64>>(&self, range: S) -> wgpu::BufferSlice {
        self.buffer.slice(range)
    }
}

pub(crate) struct Shader {
    pub(crate) module: wgpu::ShaderModule,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::RenderPipeline,
}

pub(crate) enum BindLayoutEntry {
    Texture,
}

impl Shader {
    pub fn new(
        device: &wgpu::Device,
        shader: &str,
        bind_layout_entries: &[BindLayoutEntry],
        push_constant_size: u32,
        vertex_layout: &[wgpu::VertexFormat],
    ) -> Self
    {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        let wgpu_bind_group_layout_entries =
            bind_layout_entries
                .iter()
                .enumerate()
                .map(|(index, entry)| {
                    match entry {
                        BindLayoutEntry::Texture => {
                            wgpu::BindGroupLayoutEntry {
                                binding: index as u32,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    multisampled: false,
                                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            }
                        }
                    }
                })
                .collect::<Vec<wgpu::BindGroupLayoutEntry>>();

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &wgpu_bind_group_layout_entries,
                label: None,
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::VERTEX,
                    range: 0..push_constant_size,
                }],
                label: None,
            });

        let mut vertex_stride: u64 = 0;
        let mut vertex_attributes: Vec<wgpu::VertexAttribute> = Vec::new();
        for (index, entry) in vertex_layout.iter().enumerate() {
            vertex_attributes.push(wgpu::VertexAttribute {
                offset: vertex_stride,
                format: *entry,
                shader_location: index as u32,
            });
            vertex_stride += entry.size();
        }

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: vertex_stride,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: vertex_attributes.as_slice(),
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            label: None,
        });

        Self {
            module,
            bind_group_layout,
            pipeline,
        }
    }
}

pub(crate) struct Texture {
    pub image_desc: ImageDesc,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub extent: wgpu::Extent3d,
}

impl Texture {
    pub fn new(device: &wgpu::Device, image_desc: &ImageDesc) -> Self {
        let extent = wgpu::Extent3d {
            width: image_desc.width(),
            height: image_desc.height(),
            depth_or_array_layers: 1,
        };

        let usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::from(image_desc.color_format()),
            usage,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            image_desc: image_desc.clone(),
            texture,
            view,
            extent,
        }
    }

    pub fn write(&self, queue: &wgpu::Queue, image: &Image) -> anyhow::Result<()> {
        if self.image_desc != image.desc {
            return Err(anyhow::anyhow!("image info mismatch"));
        }

        queue.write_texture(
            self.texture.as_image_copy(),
            &image.bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.image_desc.stride()),
                rows_per_image: Some(self.image_desc.height()),
            },
            self.extent,
        );

        Ok(())
    }

    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue, image: &mut Image) -> anyhow::Result<()> {
        if self.image_desc != image.desc {
            return Err(anyhow::anyhow!("image info mismatch"));
        }

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size: self.image_desc.size_in_bytes() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
            label: Some("Read buffer"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: Default::default(),
            },
            wgpu::ImageCopyBuffer {
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.image_desc.stride()),
                    rows_per_image: Some(self.image_desc.height()),
                },
            },
            self.extent,
        );

        queue.submit(Some(encoder.finish()));


        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        {
            let data = slice.get_mapped_range();
            image.bytes = data.to_vec();
            drop(data);
        }

        buffer.unmap();

        Ok(())
    }
}

