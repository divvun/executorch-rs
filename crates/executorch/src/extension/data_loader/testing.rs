//! Test-only helper mirroring extension/testing_util/temp_file.h.
//!
//! Creates and manages a named temporary file in the file system, deleting it
//! when the instance is dropped. Only for use in tests.

extern crate std;

use std::ffi::CString;
use std::io::{Seek, SeekFrom, Write};
use std::string::String;

/// Creates and manages a named temporary file in the file system. Deletes the
/// file when this instance is destroyed.
pub struct TempFile {
    path_: String,
}

impl TempFile {
    /// Creates a temporary file with the provided contents.
    pub fn new(data: &[u8]) -> Self {
        let path = Self::unique_path();
        let mut file = std::fs::File::create(&path).expect("open temp file failed");
        file.write_all(data).expect("failed to write temp file");
        TempFile { path_: path }
    }

    /// Creates a sparse temporary file with a byte slice at a specific offset.
    /// The file will have the specified total size, but only the data at the
    /// given offset is written.
    pub fn new_sparse(offset: usize, data: &[u8], file_size: usize) -> Self {
        assert!(
            file_size >= offset + data.len(),
            "file_size must be >= offset + data.len()"
        );
        let path = Self::unique_path();
        let mut file = std::fs::File::create(&path).expect("open temp file failed");

        // Seek to the offset and write the data.
        file.seek(SeekFrom::Start(offset as u64))
            .expect("failed to seek to offset");
        file.write_all(data)
            .expect("failed to write data at offset");

        // Ensure the file is the specified size.
        if file_size > offset + data.len() {
            file.seek(SeekFrom::Start((file_size - 1) as u64))
                .expect("failed to seek to file_size - 1");
            file.write_all(b"\0").expect("failed to write final byte");
        }
        drop(file);

        TempFile { path_: path }
    }

    /// The absolute path to the temporary file.
    pub fn path(&self) -> &str {
        &self.path_
    }

    /// The path as a NUL-terminated C string, for the `*const c_char` loader
    /// APIs.
    pub fn path_c(&self) -> CString {
        CString::new(self.path_.as_str()).expect("path contained a NUL byte")
    }

    // Mirrors the C++ `std::tmpnam(...) + "-executorch-testing"` unique-name
    // scheme, using the OS temp dir plus process id and an atomic counter.
    fn unique_path() -> String {
        use core::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let name = std::format!("executorch-testing-{}-{}", pid, n);
        dir.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        if !self.path_.is_empty() {
            let _ = std::fs::remove_file(&self.path_);
        }
    }
}
