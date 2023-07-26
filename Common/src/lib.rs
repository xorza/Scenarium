use std::path::Path;

#[macro_use]
pub mod macros;
pub mod scoped_ref;
pub mod apply;
pub mod toggle;
pub mod log_setup;


pub const EPSILON: f64 = 1e-10;

pub fn get_file_extension(filename: &str) -> anyhow::Result<&str> {
    let extension = Path::new(filename)
        .extension()
        .and_then(|os_str| os_str.to_str())
        .ok_or(anyhow::anyhow!("Failed to get file extension"))?;

    Ok(extension)
}

pub fn is_debug() -> bool {
    cfg!(debug_assertions)
}
