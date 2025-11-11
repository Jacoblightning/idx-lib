pub use ndarray::ArrayD;
use ndarray::IxDyn;
use std::io::{Read, Seek};
use std::convert::From;

#[derive(Debug, Default, Clone)]
pub enum IdxData {
    #[default]
    None,
    UnsignedByte(u8),
    SignedByte(i8),
    Short(i16),
    Int(i32),
    Float(f32),
    Double(f64),
}

impl IdxData {
    pub fn cast_as<T: From<u8> + From<i8> + From<i16> + From<i32> + From<f32> + From<f64>>(&self) -> Option<T> {
        match self {
            Self::None => None,
            Self::UnsignedByte(val) => Some(<u8  as Into<T>>::into(*val)),
            Self::SignedByte(val)   => Some(<i8  as Into<T>>::into(*val)),
            Self::Short(val)        => Some(<i16 as Into<T>>::into(*val)),
            Self::Int(val)          => Some(<i32 as Into<T>>::into(*val)),
            Self::Float(val)        => Some(<f32 as Into<T>>::into(*val)),
            Self::Double(val)       => Some(<f64 as Into<T>>::into(*val)),
        }
    }
}

#[derive(Debug)]
enum IdxType {
    UnsignedByte,
    SignedByte,
    Short,
    Int,
    Float,
    Double,
}

#[derive(Debug, Clone)]
struct IdxError;

impl std::fmt::Display for IdxError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error")
    }
}

impl std::error::Error for IdxError {}

fn recurser(
    idx_source: &mut (impl Read + Seek),
    data: &mut ArrayD<IdxData>,
    dimension_sizes: &[usize],
    data_type: &IdxType,
    past_idxes: &mut Vec<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let current_idx = past_idxes.len();
    // If we are on the last dimension
    if current_idx == dimension_sizes.len() {
        // Add the data
        match data_type {
            IdxType::UnsignedByte => {
                let mut buf = [0u8; 1];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::UnsignedByte(u8::from_be_bytes(buf));
            }
            IdxType::SignedByte => {
                let mut buf = [0u8; 1];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::SignedByte(i8::from_be_bytes(buf));
            }
            IdxType::Short => {
                let mut buf = [0u8; 2];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::Short(i16::from_be_bytes(buf));
            }
            IdxType::Int => {
                let mut buf = [0u8; 4];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::Int(i32::from_be_bytes(buf));
            }
            IdxType::Float => {
                let mut buf = [0u8; 4];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::Float(f32::from_be_bytes(buf));
            }
            IdxType::Double => {
                let mut buf = [0u8; 8];
                idx_source.read_exact(&mut buf)?;
                data[&past_idxes[..]] = IdxData::Double(f64::from_be_bytes(buf));
            }
        }
        Ok(())
    } else {
        let my_idx = past_idxes.len();
        past_idxes.push(0);
        // Not in the final dimension
        for i in 0..dimension_sizes[current_idx] {
            past_idxes[my_idx] = i;
            recurser(idx_source, data, dimension_sizes, data_type, past_idxes)?;
        }
        // Remember to remove our index
        past_idxes.pop();
        Ok(())
    }
}

fn process_dimensions(
    idx_source: &mut (impl Read + Seek),
    data: &mut ArrayD<IdxData>,
    dimension_sizes: &[usize],
    data_type: &IdxType,
) -> Result<(), Box<dyn std::error::Error>> {
    recurser(idx_source, data, dimension_sizes, data_type, &mut vec![])
}

pub fn read_idx(
    idx_source: &mut (impl Read + Seek),
) -> Result<ArrayD<IdxData>, Box<dyn std::error::Error>> {
    // First 2 bytes are always 0
    idx_source.seek_relative(2)?;

    // Data in idx is stored in big endian format
    let mut data_type_buf = [0u8; 1];
    let mut dimension_count_buf = [0u8; 1];

    idx_source.read_exact(&mut data_type_buf)?;
    idx_source.read_exact(&mut dimension_count_buf)?;

    let data_type = match u8::from_be_bytes(data_type_buf) {
        0x08u8 => IdxType::UnsignedByte,
        0x09u8 => IdxType::SignedByte,
        0x0Bu8 => IdxType::Short,
        0x0Cu8 => IdxType::Int,
        0x0Du8 => IdxType::Float,
        0x0Eu8 => IdxType::Double,
        _ => return Err(IdxError.into()),
    };
    let dimension_count = u8::from_be_bytes(dimension_count_buf);

    let mut dimension_sizes = Vec::with_capacity(dimension_count as usize);

    for _ in 0..dimension_count {
        let mut dimension_size_buf = [0u8; 4];
        idx_source.read_exact(&mut dimension_size_buf)?;
        dimension_sizes.push(i32::from_be_bytes(dimension_size_buf) as usize);
    }

    //println!("Data Type: {:#?}\nDimension Count: {}\nDimension Sizes: {:#?}", data_type, dimension_count, dimension_sizes);

    let mut data: ArrayD<IdxData> = ArrayD::default(IxDyn(&dimension_sizes));

    process_dimensions(idx_source, &mut data, &dimension_sizes, &data_type)?;

    Ok(data)
}
