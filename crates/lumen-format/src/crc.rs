//! Minimal CRC32 (IEEE polynomial) for LBC header/index checksums.
//!
//! Hand-rolled with a lookup table — zero external dependencies.

const POLYNOMIAL: u32 = 0xEDB8_8320; // IEEE reversed

/// Build the 256-entry CRC32 lookup table at compile time.
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = build_table();

/// Compute CRC32 (IEEE) over a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    crc32_finalize(crc32_update(CRC32_INIT, data))
}

/// Initial CRC32 accumulator state.
pub const CRC32_INIT: u32 = 0xFFFF_FFFFu32;

/// Feed additional bytes into an ongoing CRC32 computation.
pub fn crc32_update(state: u32, data: &[u8]) -> u32 {
    let mut crc = state;
    for &byte in data {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    crc
}

/// Finalize a CRC32 computation.
pub fn crc32_finalize(state: u32) -> u32 {
    state ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // "123456789" => 0xCBF43926 (standard CRC32 test vector)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn empty_input() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn single_byte() {
        // 'A' => known value
        assert_eq!(crc32(b"A"), 0xD3D9_9E8B);
    }
}
