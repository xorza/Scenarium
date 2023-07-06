use std::fs::File;
use std::path::Path;

use image_lib::ImageFormat;
use tiff::decoder::DecodingResult;

use crate::color_format::*;
use crate::image_convertion::convert_image;
use crate::tiff_extentions::save_tiff;

fn get_file_extension(filename: &str) -> anyhow::Result<&str> {
    let extension = Path::new(filename)
        .extension()
        .and_then(|os_str| os_str.to_str())
        .ok_or(anyhow::anyhow!("Failed to get file extension"))?;

    Ok(extension)
}

fn align_stride(n: u32) -> u32 {
    // align to 4
    // (n + 3) & !3

    // align to 2
    // if n % 2 == 0 {
    //     n
    // } else {
    //     n + 1
    // }

    n
}


#[derive(Clone)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub color_format: ColorFormat,
}

#[derive(Clone)]
pub struct Image {
    pub info: ImageInfo,
    pub bytes: Vec<u8>,
}

impl Image {
    pub fn new_empty(
        width: u32,
        height: u32,
        color_format: ColorFormat, )
        -> anyhow::Result<Image>
    {
        color_format.validate()?;

        let info = ImageInfo::new(width, height, color_format);
        let bytes = vec![0; (info.stride * height) as usize];

        Ok(Image {
            info,
            bytes,
        })
    }

    pub fn new_with_data(
        width: u32,
        height: u32,
        color_format: ColorFormat,
        bytes: Vec<u8>)
        -> anyhow::Result<Image>
    {
        color_format.validate()?;

        let info = ImageInfo::new(width, height, color_format);

        Ok(Image {
            info,
            bytes,
        })
    }

    pub fn read_file(filename: &str) -> anyhow::Result<Image> {
        let extension = get_file_extension(filename)?;

        let image =
            match extension {
                "png" | "jpeg" | "jpg" => Image::load_png_jpeg(filename)?,
                "tiff" => Image::load_tiff(filename)?,

                _ => return Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
            };

        Ok(image)
    }

    fn load_png_jpeg(filename: &str) -> anyhow::Result<Image> {
        let img =
            image_lib::open(filename)
                .expect("Failed to open image");

        let (channel_count, channel_size, channel_type) = match img.color() {
            // @formatter:off
            image_lib::ColorType::L8      => (ChannelCount::Gray,      ChannelSize:: _8bit, ChannelType::UInt  ),
            image_lib::ColorType::L16     => (ChannelCount::Gray,      ChannelSize::_16bit, ChannelType::UInt  ),
            image_lib::ColorType::La8     => (ChannelCount::GrayAlpha, ChannelSize:: _8bit, ChannelType::UInt  ),
            image_lib::ColorType::La16    => (ChannelCount::GrayAlpha, ChannelSize::_16bit, ChannelType::UInt  ),
            image_lib::ColorType::Rgb8    => (ChannelCount::Rgb,       ChannelSize:: _8bit, ChannelType::UInt  ),
            image_lib::ColorType::Rgb16   => (ChannelCount::Rgb,       ChannelSize::_16bit, ChannelType::UInt  ),
            image_lib::ColorType::Rgba8   => (ChannelCount::Rgba,      ChannelSize:: _8bit, ChannelType::UInt  ),
            image_lib::ColorType::Rgba16  => (ChannelCount::Rgba,      ChannelSize::_16bit, ChannelType::UInt  ),
            image_lib::ColorType::Rgb32F  => (ChannelCount::Rgb,       ChannelSize::_32bit, ChannelType::Float ),
            image_lib::ColorType::Rgba32F => (ChannelCount::Rgba,      ChannelSize::_32bit, ChannelType::Float ),
            _ =>  panic!("Unsupported color type: {:?}", img.color()),
            // @formatter:on
        };

        let bytes = img.as_bytes().to_vec();

        let image = Image {
            info: ImageInfo::new(
                img.width(),
                img.height(),
                ColorFormat::from((channel_count, channel_size, channel_type)), ),
            bytes,
        };

        Ok(image)
    }
    fn load_tiff(filename: &str) -> anyhow::Result<Image> {
        let mut decoder = tiff::decoder::Decoder::new(
            File::open(filename)?
        )?;

        let (channel_bits, channel_count) = match decoder.colortype()? {
            // @formatter:off
            tiff::ColorType::Gray  (b) => (b, ChannelCount::Gray      ),
            tiff::ColorType::GrayA (b) => (b, ChannelCount::GrayAlpha ),
            tiff::ColorType::RGB   (b) => (b, ChannelCount::Rgb      ),
            tiff::ColorType::RGBA  (b) => (b, ChannelCount::Rgba      ),
            _ => panic!("Unsupported color type: {:?}", decoder.colortype()?),
            // @formatter:on
        };

        let img = decoder.read_image()?;
        let bytes: Vec<u8> = match &img {
            // @formatter:off
            DecodingResult::U8 (buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::I8 (buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::U16(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::I16(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::U32(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::I32(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::U64(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::I64(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::F32(buf) => bytemuck::cast_slice(buf).to_vec(),
            DecodingResult::F64(buf) => bytemuck::cast_slice(buf).to_vec(),
            // @formatter:on
        };

        let channel_type = match &img {
            // @formatter:off
            DecodingResult::U8 (_) => ChannelType::UInt,
            DecodingResult::U16(_) => ChannelType::UInt,
            DecodingResult::U32(_) => ChannelType::UInt,
            DecodingResult::U64(_) => ChannelType::UInt,
            DecodingResult::I8 (_) => ChannelType::Int,
            DecodingResult::I16(_) => ChannelType::Int,
            DecodingResult::I32(_) => ChannelType::Int,
            DecodingResult::I64(_) => ChannelType::Int,
            DecodingResult::F32(_) => ChannelType::Float,
            DecodingResult::F64(_) => ChannelType::Float,
            // @formatter:on
        };

        let (w, h) = decoder.dimensions()?;

        let channel_size = ChannelSize::from_bit_count(channel_bits as u32);
        let color_format = ColorFormat::from((channel_count, channel_size, channel_type));

        let image = Image {
            info: ImageInfo {
                width: w,
                height: h,
                stride: bytes.len() as u32 / h,
                color_format,
            },
            bytes,
        };

        Ok(image)
    }


    pub fn save_file(&self, filename: &str) -> anyhow::Result<()> {
        let extension = get_file_extension(filename)?;

        match extension {
            "png" => self.save_png(filename)?,
            "jpeg" | "jpg" => self.save_jpg(filename)?,
            "tiff" => self.save_tiff(filename)?,

            _ => return Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
        };

        Ok(())
    }

    fn save_jpg(&self, filename: &str) -> anyhow::Result<()> {
        if self.info.color_format.channel_type != ChannelType::UInt {
            return Err(anyhow::anyhow!("Unsupported JPEG channel type: {:?}", self.info.color_format.channel_type));
        }

        let color_format = match self.info.color_format.channel_size {
            ChannelSize::_8bit => match self.info.color_format.channel_count {
                ChannelCount::Gray => image_lib::ColorType::L8,
                ChannelCount::Rgb => image_lib::ColorType::Rgb8,

                _ => return Err(anyhow::anyhow!("Unsupported JPEG color format: {:?}", self.info.color_format.channel_count)),
            },

            _ => return Err(anyhow::anyhow!("Unsupported JPEG channel size: {:?}", self.info.color_format.channel_size)),
        };

        image_lib::save_buffer_with_format(
            filename,
            &self.bytes,
            self.info.width,
            self.info.height,
            color_format,
            ImageFormat::Jpeg,
        )?;

        Ok(())
    }
    fn save_png(&self, filename: &str) -> anyhow::Result<()> {
        if self.info.color_format.channel_type != ChannelType::UInt {
            return Err(anyhow::anyhow!("Unsupported PNG channel type: {:?}", self.info.color_format.channel_type));
        }

        let color_format = match self.info.color_format.channel_size {
            ChannelSize::_8bit => match self.info.color_format.channel_count {
                ChannelCount::Gray => image_lib::ColorType::L8,
                ChannelCount::GrayAlpha => image_lib::ColorType::La8,
                ChannelCount::Rgb => image_lib::ColorType::Rgb8,
                ChannelCount::Rgba => image_lib::ColorType::Rgba8,
            },
            ChannelSize::_16bit => match self.info.color_format.channel_count {
                ChannelCount::Gray => image_lib::ColorType::L16,
                ChannelCount::GrayAlpha => image_lib::ColorType::La16,
                ChannelCount::Rgb => image_lib::ColorType::Rgb16,
                ChannelCount::Rgba => image_lib::ColorType::Rgba16,
            },

            _ => return Err(anyhow::anyhow!("Unsupported PNG channel size: {:?}", self.info.color_format.channel_size)),
        };

        image_lib::save_buffer_with_format(
            filename,
            &self.bytes,
            self.info.width,
            self.info.height,
            color_format,
            ImageFormat::Png,
        )?;

        Ok(())
    }
    fn save_tiff(&self, filename: &str) -> anyhow::Result<()> {
        save_tiff(self, filename)
    }


    pub fn convert(
        &self,
        color_format: ColorFormat, )
        -> anyhow::Result<Image>
    {
        color_format.validate()?;

        if self.info.color_format == color_format {
            return Err(anyhow::anyhow!("Image is already in the requested format"));
        }

        let mut result = Image::new_empty(
            self.info.width,
            self.info.height,
            color_format,
        )?;

        convert_image(self, &mut result)?;

        Ok(result)
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        self.info.color_format.byte_count()
    }
}


impl ImageInfo {
    pub fn new(width: u32,
               height: u32,
               color_format: ColorFormat, ) -> Self {
        let stride = align_stride(width * color_format.channel_count as u32 * color_format.channel_size as u32);

        Self {
            width,
            height,
            stride,
            color_format,
        }
    }
}
