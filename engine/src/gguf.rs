//! The GGUF file format (version 3): llama.cpp's container for a model's
//! metadata, tokenizer and tensors in one file.
//!
//! Layout, all little-endian: the magic `GGUF`, a `u32` version, a `u64`
//! tensor count and a `u64` metadata count; the metadata as key/value
//! pairs; one descriptor per tensor (name, shape, type, offset); then,
//! aligned to `general.alignment` (32 by default), the tensor data, which
//! each descriptor's offset is relative to.
//!
//! Nothing is copied: strings, arrays and tensors are views into the
//! file's bytes, which the caller keeps in memory.

use alloc::vec::Vec;

/// Why a file did not parse as GGUF.
#[derive(Debug, PartialEq, Eq)]
pub enum GgufError {
    /// Not a GGUF file: the first four bytes are not `GGUF`.
    BadMagic,
    /// A GGUF version this parser does not read.
    Version(u32),
    /// The file ends inside the structure being read.
    Truncated,
    /// A metadata value of a type this parser does not know.
    ValueType(u32),
    /// A string that is not UTF-8.
    NotUtf8,
    /// A tensor's data lies outside the file.
    TensorBounds,
    /// A Q8_0 tensor cannot be repacked: its rows do not come in fours, or
    /// are not whole blocks. The index is the tensor's, in file order.
    NotRepackable(usize),
}

/// A metadata value's type, as GGUF numbers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueType {
    /// `u8`.
    U8,
    /// `i8`.
    I8,
    /// `u16`.
    U16,
    /// `i16`.
    I16,
    /// `u32`.
    U32,
    /// `i32`.
    I32,
    /// `f32`.
    F32,
    /// A one-byte boolean.
    Bool,
    /// A `u64`-length-prefixed UTF-8 string.
    String,
    /// A typed array.
    Array,
    /// `u64`.
    U64,
    /// `i64`.
    I64,
    /// `f64`.
    F64,
}

impl ValueType {
    fn from_u32(value: u32) -> Result<Self, GgufError> {
        Ok(match value {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            other => return Err(GgufError::ValueType(other)),
        })
    }

    /// Bytes in one value of a fixed-size type; `None` for strings and
    /// arrays.
    fn fixed_size(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

/// A metadata value.
#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    /// An unsigned integer of any width.
    Uint(u64),
    /// A signed integer of any width.
    Int(i64),
    /// A float of either width.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A string.
    String(&'a str),
    /// An array, read element by element through [`Array`].
    Array(Array<'a>),
}

/// An array value: its element type, its length, and its bytes.
#[derive(Clone, Copy, Debug)]
pub struct Array<'a> {
    /// The elements' type.
    pub element: ValueType,
    /// How many there are.
    pub len: usize,
    /// The encoded elements.
    bytes: &'a [u8],
}

impl<'a> Array<'a> {
    /// The elements as strings, if they are strings.
    pub fn strings(self) -> Option<impl Iterator<Item = &'a str> + 'a> {
        if self.element != ValueType::String {
            return None;
        }
        let mut reader = Reader {
            bytes: self.bytes,
            at: 0,
        };
        let len = self.len;
        Some((0..len).map_while(move |_| reader.string().ok()))
    }

    /// The elements as `i32`s, if they are.
    pub fn i32s(self) -> Option<impl Iterator<Item = i32> + 'a> {
        if self.element != ValueType::I32 {
            return None;
        }
        Some(
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| i32::from_le_bytes(b)),
        )
    }
}

/// A tensor's element type, as GGUF numbers the ones this crate reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    /// 32-bit float.
    F32,
    /// Blocks of 32 `i8`s sharing one `f16` scale. See [`crate::quant`].
    Q8_0,
    /// Anything else, by its GGUF number.
    Other(u32),
}

/// A tensor: its name, shape and type, and its data.
#[derive(Clone, Copy, Debug)]
pub struct Tensor<'a> {
    /// The tensor's name, such as `blk.0.attn_q.weight`.
    pub name: &'a str,
    /// The shape, innermost dimension first, as ggml orders it: a matrix
    /// of `rows` rows of `cols` columns is `[cols, rows]`.
    pub dims: [u64; 4],
    /// How many of `dims` are meaningful.
    pub n_dims: usize,
    /// The element type.
    pub kind: TensorType,
    /// The tensor's bytes.
    pub data: &'a [u8],
}

impl Tensor<'_> {
    /// Elements in the tensor.
    pub fn elements(&self) -> u64 {
        self.dims[..self.n_dims].iter().product()
    }
}

/// A parsed GGUF file.
pub struct Gguf<'a> {
    metadata: Vec<(&'a str, Value<'a>)>,
    tensors: Vec<Tensor<'a>>,
}

/// The data section's alignment when `general.alignment` does not say.
const DEFAULT_ALIGNMENT: u64 = 32;

impl<'a> Gguf<'a> {
    /// Parses `file`.
    pub fn parse(file: &'a [u8]) -> Result<Self, GgufError> {
        let mut reader = Reader { bytes: file, at: 0 };
        if reader.take(4)? != b"GGUF" {
            return Err(GgufError::BadMagic);
        }
        let version = reader.u32()?;
        if version != 3 {
            return Err(GgufError::Version(version));
        }
        let tensor_count = reader.u64()? as usize;
        let metadata_count = reader.u64()? as usize;

        let mut metadata = Vec::with_capacity(metadata_count);
        for _ in 0..metadata_count {
            let key = reader.string()?;
            let kind = ValueType::from_u32(reader.u32()?)?;
            metadata.push((key, reader.value(kind)?));
        }

        // Descriptors first, data offsets resolved once the data section's
        // start is known.
        let mut descriptors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = reader.string()?;
            let n_dims = reader.u32()? as usize;
            if n_dims > 4 {
                return Err(GgufError::TensorBounds);
            }
            let mut dims = [1u64; 4];
            for dim in dims.iter_mut().take(n_dims) {
                *dim = reader.u64()?;
            }
            let kind = match reader.u32()? {
                0 => TensorType::F32,
                8 => TensorType::Q8_0,
                other => TensorType::Other(other),
            };
            let offset = reader.u64()?;
            descriptors.push((name, dims, n_dims, kind, offset));
        }

        let alignment = metadata
            .iter()
            .find(|(key, _)| *key == "general.alignment")
            .and_then(|(_, value)| match value {
                Value::Uint(v) => Some(*v),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);
        let data_start = (reader.at as u64).next_multiple_of(alignment) as usize;

        let mut tensors = Vec::with_capacity(tensor_count);
        for (name, dims, n_dims, kind, offset) in descriptors {
            let elements: u64 = dims[..n_dims].iter().product();
            let bytes = match kind {
                TensorType::F32 => elements * 4,
                TensorType::Q8_0 => elements / 32 * 34,
                // Unknown types get no data; using one is an error at the
                // point a model asks for it.
                TensorType::Other(_) => 0,
            } as usize;
            let start = data_start + offset as usize;
            let data = file
                .get(start..start + bytes)
                .ok_or(GgufError::TensorBounds)?;
            tensors.push(Tensor {
                name,
                dims,
                n_dims,
                kind,
                data,
            });
        }

        Ok(Self { metadata, tensors })
    }

    /// The metadata value under `key`.
    pub fn get(&self, key: &str) -> Option<Value<'a>> {
        self.metadata
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, value)| *value)
    }

    /// An unsigned integer value.
    pub fn uint(&self, key: &str) -> Option<u64> {
        match self.get(key)? {
            Value::Uint(v) => Some(v),
            Value::Int(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// A float value.
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Value::Float(v) => Some(v),
            _ => None,
        }
    }

    /// A string value.
    pub fn string(&self, key: &str) -> Option<&'a str> {
        match self.get(key)? {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// An array value.
    pub fn array(&self, key: &str) -> Option<Array<'a>> {
        match self.get(key)? {
            Value::Array(array) => Some(array),
            _ => None,
        }
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Option<&Tensor<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Every tensor, in file order.
    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    /// Repacks every Q8_0 tensor in `file` in place for computing with
    /// (see [`crate::quant`]), then parses it.
    ///
    /// Only what this returns can be handed to
    /// [`crate::model::Weights::from_gguf`], so a model cannot be built
    /// over blocks still in the file's layout: they would be read as
    /// garbage, not rejected.
    ///
    /// Repacking needs scratch space the size of the largest Q8_0 tensor,
    /// for as long as it takes to rearrange that tensor.
    pub fn parse_repacked(file: &'a mut [u8]) -> Result<Repacked<'a>, GgufError> {
        // Each Q8_0 tensor's place in the file, its row width, and its index.
        let tensors: Vec<(usize, usize, usize, usize)> = Gguf::parse(file)?
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.kind == TensorType::Q8_0)
            .map(|(i, t)| {
                let start = t.data.as_ptr() as usize - file.as_ptr() as usize;
                (start, start + t.data.len(), t.dims[0] as usize, i)
            })
            .collect();
        for (start, end, cols, index) in tensors {
            if !crate::quant::repack(&mut file[start..end], cols) {
                return Err(GgufError::NotRepackable(index));
            }
        }
        let file: &'a [u8] = file;
        Ok(Repacked(Gguf::parse(file)?))
    }
}

/// A GGUF file whose Q8_0 tensors [`Gguf::parse_repacked`] has repacked.
/// Everything else reads as through the [`Gguf`] it dereferences to.
pub struct Repacked<'a>(Gguf<'a>);

impl<'a> core::ops::Deref for Repacked<'a> {
    type Target = Gguf<'a>;

    fn deref(&self) -> &Gguf<'a> {
        &self.0
    }
}

/// A cursor over the file's bytes.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.at.checked_add(n).ok_or(GgufError::Truncated)?;
        let slice = self.bytes.get(self.at..end).ok_or(GgufError::Truncated)?;
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<&'a str, GgufError> {
        let len = self.u64()? as usize;
        core::str::from_utf8(self.take(len)?).map_err(|_| GgufError::NotUtf8)
    }

    /// Reads one value of type `kind`.
    fn value(&mut self, kind: ValueType) -> Result<Value<'a>, GgufError> {
        let fixed = |r: &mut Self, n| r.take(n).map(|b: &[u8]| b);
        Ok(match kind {
            ValueType::U8 => Value::Uint(u64::from(fixed(self, 1)?[0])),
            ValueType::I8 => Value::Int(i64::from(fixed(self, 1)?[0] as i8)),
            ValueType::U16 => Value::Uint(u64::from(u16::from_le_bytes(
                fixed(self, 2)?.try_into().unwrap(),
            ))),
            ValueType::I16 => Value::Int(i64::from(i16::from_le_bytes(
                fixed(self, 2)?.try_into().unwrap(),
            ))),
            ValueType::U32 => Value::Uint(u64::from(self.u32()?)),
            ValueType::I32 => Value::Int(i64::from(self.u32()? as i32)),
            ValueType::F32 => Value::Float(f64::from(f32::from_bits(self.u32()?))),
            ValueType::Bool => Value::Bool(fixed(self, 1)?[0] != 0),
            ValueType::String => Value::String(self.string()?),
            ValueType::U64 => Value::Uint(self.u64()?),
            ValueType::I64 => Value::Int(self.u64()? as i64),
            ValueType::F64 => Value::Float(f64::from_bits(self.u64()?)),
            ValueType::Array => {
                let element = ValueType::from_u32(self.u32()?)?;
                let len = self.u64()? as usize;
                let start = self.at;
                match element.fixed_size() {
                    Some(size) => {
                        self.take(len.checked_mul(size).ok_or(GgufError::Truncated)?)?;
                    }
                    None if element == ValueType::String => {
                        for _ in 0..len {
                            self.string()?;
                        }
                    }
                    // Arrays of arrays: no model this crate reads has one.
                    None => return Err(GgufError::ValueType(9)),
                }
                Value::Array(Array {
                    element,
                    len,
                    bytes: &self.bytes[start..self.at],
                })
            }
        })
    }
}
