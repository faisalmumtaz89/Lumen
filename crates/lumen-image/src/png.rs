//! A minimal PNG writer and reader for 8-bit RGBA images.
//!
//! Written rather than depended on: the workspace has no image codec, the only
//! format needed is non-interlaced 8-bit RGBA, and a dependency for that is more
//! surface than the ~200 lines it replaces.
//!
//! Compression is zlib with stored (uncompressed) deflate blocks. That is valid
//! PNG that every decoder reads; it costs file size, not correctness, and it
//! keeps this file auditable. The reader accepts stored blocks, which is what
//! the writer emits and what the round-trip test checks.

/// Errors from encoding or decoding.
#[derive(Debug)]
pub enum PngError {
    Io(std::io::Error),
    NotPng,
    /// A feature outside the subset this module handles.
    Unsupported(String),
    Truncated,
    /// The deflate stream or a chunk's checksum is inconsistent.
    Corrupt(String),
}

impl std::fmt::Display for PngError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::NotPng => write!(f, "not a PNG file"),
            Self::Unsupported(w) => write!(f, "unsupported: {w}"),
            Self::Truncated => write!(f, "file ends mid-structure"),
            Self::Corrupt(w) => write!(f, "corrupt: {w}"),
        }
    }
}

impl std::error::Error for PngError {}

impl From<std::io::Error> for PngError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// An 8-bit RGBA image, row-major, four bytes per pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct Rgba {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgba {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Self {
        assert_eq!(width * height * 4, data.len(), "pixel buffer size mismatch");
        Self {
            width,
            height,
            data,
        }
    }

    /// Take an image from the VAE's output.
    ///
    /// The decoder emits planar sample-major data — four full planes in the
    /// order R, G, B, A, each `height * width` long — so the channel is the
    /// OUTER axis and the pixel position inner. Reading that buffer as
    /// interleaved quads instead would scatter all four channels across the
    /// wrong pixels, which still produces a correctly-sized image of the right
    /// colour range, so the transpose is done here where the layout changes
    /// rather than left to each caller.
    ///
    /// Values are clamped to [-1, 1] and rescaled to [0, 255], the transform the
    /// reference's postprocessor applies.
    pub fn from_planar_rgba(width: usize, height: usize, values: &[f32]) -> Self {
        let plane = width * height;
        assert_eq!(
            plane * 4,
            values.len(),
            "planar buffer size mismatch: expected 4 planes of {plane}"
        );
        let mut data = vec![0u8; plane * 4];
        for c in 0..4 {
            let src = &values[c * plane..(c + 1) * plane];
            for (i, &v) in src.iter().enumerate() {
                let clamped = v.clamp(-1.0, 1.0);
                data[i * 4 + c] = (((clamped + 1.0) * 0.5) * 255.0).round() as u8;
            }
        }
        Self {
            width,
            height,
            data,
        }
    }
}

fn crc_table() -> &'static [u32; 256] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (n, slot) in t.iter_mut().enumerate() {
            let mut c = n as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
            *slot = c;
        }
        t
    })
}

fn crc32(bytes: &[u8]) -> u32 {
    let t = crc_table();
    let mut c = 0xFFFF_FFFFu32;
    for &b in bytes {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    let mut body = Vec::with_capacity(4 + payload.len());
    body.extend_from_slice(kind);
    body.extend_from_slice(payload);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// Wrap raw bytes in a zlib stream of stored deflate blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // zlib header, no preset dictionary
                                    // A stored block carries a 16-bit length, so it caps at 65535 bytes, and
                                    // the very last one is flagged with BFINAL.
    let mut pos = 0usize;
    if raw.is_empty() {
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0xFF, 0xFF]);
    }
    while pos < raw.len() {
        let take = (raw.len() - pos).min(65535);
        let last = pos + take == raw.len();
        out.push(if last { 0x01 } else { 0x00 });
        out.extend_from_slice(&(take as u16).to_le_bytes());
        out.extend_from_slice(&(!(take as u16)).to_le_bytes());
        out.extend_from_slice(&raw[pos..pos + take]);
        pos += take;
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// Encode an 8-bit RGBA image as a non-interlaced PNG.
pub fn encode(image: &Rgba) -> Vec<u8> {
    let mut out = Vec::with_capacity(image.data.len() / 2 + 128);
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(image.width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(image.height as u32).to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // colour type: truecolour with alpha
    ihdr.push(0); // deflate
    ihdr.push(0); // adaptive filtering
    ihdr.push(0); // no interlace
    chunk(&mut out, b"IHDR", &ihdr);

    // Each scanline is prefixed with its filter byte; filter 0 (None) keeps the
    // encoder trivial and the output exact.
    let stride = image.width * 4;
    let mut raw = Vec::with_capacity((stride + 1) * image.height);
    for row in 0..image.height {
        raw.push(0);
        raw.extend_from_slice(&image.data[row * stride..(row + 1) * stride]);
    }
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

/// Decode a PNG produced by [`encode`], or any non-interlaced 8-bit RGBA PNG
/// whose deflate stream uses stored blocks.
pub fn decode(bytes: &[u8]) -> Result<Rgba, PngError> {
    if bytes.len() < 8 || bytes[..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        return Err(PngError::NotPng);
    }
    let mut pos = 8usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut idat: Vec<u8> = Vec::new();
    while pos + 8 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let start = pos + 8;
        let end = start.checked_add(len).ok_or(PngError::Truncated)?;
        if end + 4 > bytes.len() {
            return Err(PngError::Truncated);
        }
        let payload = &bytes[start..end];
        let want = u32::from_be_bytes(bytes[end..end + 4].try_into().unwrap());
        let mut body = Vec::with_capacity(4 + len);
        body.extend_from_slice(kind);
        body.extend_from_slice(payload);
        if crc32(&body) != want {
            return Err(PngError::Corrupt(format!(
                "chunk {} checksum",
                String::from_utf8_lossy(kind)
            )));
        }
        match kind {
            b"IHDR" => {
                if len != 13 {
                    return Err(PngError::Corrupt("IHDR length".into()));
                }
                width = u32::from_be_bytes(payload[0..4].try_into().unwrap()) as usize;
                height = u32::from_be_bytes(payload[4..8].try_into().unwrap()) as usize;
                if payload[8] != 8 || payload[9] != 6 {
                    return Err(PngError::Unsupported(format!(
                        "bit depth {} colour type {}",
                        payload[8], payload[9]
                    )));
                }
                if payload[12] != 0 {
                    return Err(PngError::Unsupported("interlaced".into()));
                }
            }
            b"IDAT" => idat.extend_from_slice(payload),
            b"IEND" => break,
            _ => {} // ancillary chunks are skipped
        }
        pos = end + 4;
    }
    if width == 0 || height == 0 {
        return Err(PngError::Corrupt("no IHDR".into()));
    }
    let raw = inflate_stored(&idat)?;
    let stride = width * 4;
    if raw.len() != (stride + 1) * height {
        return Err(PngError::Corrupt(format!(
            "decompressed {} bytes for {}x{}",
            raw.len(),
            width,
            height
        )));
    }
    let mut data = Vec::with_capacity(stride * height);
    for row in 0..height {
        let off = row * (stride + 1);
        if raw[off] != 0 {
            return Err(PngError::Unsupported(format!(
                "scanline filter {}",
                raw[off]
            )));
        }
        data.extend_from_slice(&raw[off + 1..off + 1 + stride]);
    }
    Ok(Rgba {
        width,
        height,
        data,
    })
}

/// Inflate a zlib stream built from stored deflate blocks.
fn inflate_stored(z: &[u8]) -> Result<Vec<u8>, PngError> {
    if z.len() < 2 {
        return Err(PngError::Truncated);
    }
    // Only the "no compression" method is accepted.
    if z[0] & 0x0F != 8 {
        return Err(PngError::Unsupported("zlib compression method".into()));
    }
    let mut pos = 2usize;
    let mut out = Vec::new();
    loop {
        if pos >= z.len() {
            return Err(PngError::Truncated);
        }
        let header = z[pos];
        let final_block = header & 1 == 1;
        let btype = (header >> 1) & 3;
        if btype != 0 {
            return Err(PngError::Unsupported(format!("deflate block type {btype}")));
        }
        pos += 1;
        if pos + 4 > z.len() {
            return Err(PngError::Truncated);
        }
        let len = u16::from_le_bytes([z[pos], z[pos + 1]]) as usize;
        let nlen = u16::from_le_bytes([z[pos + 2], z[pos + 3]]);
        if nlen != !(len as u16) {
            return Err(PngError::Corrupt("stored block length".into()));
        }
        pos += 4;
        if pos + len > z.len() {
            return Err(PngError::Truncated);
        }
        out.extend_from_slice(&z[pos..pos + len]);
        pos += len;
        if final_block {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(w: usize, h: usize) -> Rgba {
        let data = (0..w * h * 4).map(|i| (i * 7 % 256) as u8).collect();
        Rgba::new(w, h, data)
    }

    /// The encode/decode round trip must be pixel-exact.
    #[test]
    fn round_trip_is_pixel_exact() {
        for (w, h) in [(1, 1), (3, 2), (17, 5), (64, 64)] {
            let img = sample(w, h);
            let png = encode(&img);
            let back = decode(&png).expect("decode");
            assert_eq!(back, img, "{w}x{h} round trip changed pixels");
        }
    }

    /// A body larger than one stored block must still round-trip: a stored block
    /// carries a 16-bit length, so this exercises the split.
    #[test]
    fn round_trip_across_several_stored_blocks() {
        let img = sample(300, 300); // 360,000 bytes of raw scanlines
        let png = encode(&img);
        let back = decode(&png).expect("decode");
        assert_eq!(back, img);
    }

    #[test]
    fn header_is_a_valid_png_signature() {
        let png = encode(&sample(2, 2));
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IHDR follows the signature.
        assert_eq!(&png[12..16], b"IHDR");
    }

    /// A flipped byte must be caught by the chunk checksum, not silently read.
    #[test]
    fn a_corrupted_chunk_is_rejected() {
        let mut png = encode(&sample(4, 4));
        let mid = png.len() / 2;
        png[mid] ^= 0xFF;
        assert!(decode(&png).is_err(), "a corrupted file must not decode");
    }

    #[test]
    fn a_truncated_file_is_rejected() {
        let png = encode(&sample(4, 4));
        for cut in [0, 4, 20, png.len() / 2] {
            assert!(decode(&png[..cut]).is_err(), "truncation at {cut} accepted");
        }
    }

    #[test]
    fn a_non_png_is_rejected() {
        assert!(matches!(decode(b"not a png at all"), Err(PngError::NotPng)));
    }

    /// The planar-to-interleaved transform, pinned to values that distinguish
    /// the layouts: a buffer whose channel planes hold different constants must
    /// come out as RGBA quads, not as a flat read.
    #[test]
    fn planar_input_becomes_interleaved_rgba() {
        // 2 pixels, 4 channels: planes are [R1 R2][G1 G2][B1 B2][A1 A2].
        let vals = [
            -1.0, 1.0, /*G*/ -1.0, 1.0, /*B*/ -1.0, 1.0, /*A*/ 1.0, -1.0,
        ];
        let img = Rgba::from_planar_rgba(2, 1, &vals);
        // pixel 0 = (R=-1, G=-1, B=-1, A=1) -> (0, 0, 0, 255)
        // pixel 1 = (R=1, G=1, B=1, A=-1) -> (255, 255, 255, 0)
        assert_eq!(img.data, vec![0, 0, 0, 255, 255, 255, 255, 0]);
    }

    /// The clamp and the [-1,1] -> [0,255] map.
    #[test]
    fn planar_conversion_clamps_and_scales() {
        // One pixel, planes R=0 G=1 B=-1 A=5 (out of range).
        let img = Rgba::from_planar_rgba(1, 1, &[0.0, 1.0, -1.0, 5.0]);
        assert_eq!(img.data, vec![128, 255, 0, 255]);
    }
}
