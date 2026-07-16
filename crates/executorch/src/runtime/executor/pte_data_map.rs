//! Literal port of runtime/executor/pte_data_map.cpp + runtime/executor/pte_data_map.h.
//!
//! NAME-MAPPING DEVIATION: the C++ reads the named-data segments through the
//! generated flatbuffer accessors `named_data_->size()`/`->Get(i)`,
//! `item->key()->size()`/`->data()`/`->c_str()`, `item->segment_index()`,
//! `segments_->Get(i)->offset()`/`->size()`. The Rust flatbuffers crate
//! (`crate::schema::generated::executorch_flatbuffer`) exposes snake_case,
//! usize-based accessors instead: `Vector::len()` / `Vector::get(i)` (returns
//! the element by value, non-nullable), `NamedData::key()` (returns
//! `Option<&str>`), `NamedData::segment_index()`, `DataSegment::offset()`,
//! `DataSegment::size()`. Those name/shape differences are recorded here once
//! and used verbatim below.

use crate::runtime::core::data_loader::{DataLoader, SegmentInfo, Type};
use crate::runtime::core::error::Error;
use crate::runtime::core::freeable_buffer::FreeableBuffer;
use crate::runtime::core::named_data_map::NamedDataMap;
use crate::runtime::core::result::Result;
use crate::runtime::core::tensor_layout::TensorLayout;
use crate::schema::generated::executorch_flatbuffer::{DataSegment, NamedData};
use flatbuffers::{ForwardsUOffset, Vector};

// PORT-NOTE: the C++ `flatbuffers::FlatbufferNamedData` /
// `FlatbufferDataSegment` are `flatbuffers::Vector<Offset<NamedData>>` /
// `Vector<Offset<DataSegment>>`. The Rust equivalents are the lifetime-bound
// `Vector<'a, ForwardsUOffset<NamedData<'a>>>` /
// `Vector<'a, ForwardsUOffset<DataSegment<'a>>>`, which are themselves fat
// (slice+offset) non-owning views — matching the borrowed `const Vector*` the
// C++ stored. Local type aliases mirror the C++ `using` names.
type FlatbufferNamedData<'a> = Vector<'a, ForwardsUOffset<NamedData<'a>>>;
type FlatbufferDataSegment<'a> = Vector<'a, ForwardsUOffset<DataSegment<'a>>>;

/// A NamedDataMap implementation for Flatbuffer-serialized named data
/// originating from a PTE file.
// [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map]
//
// PORT-NOTE: `loader_` is a borrowed, non-owning `DataLoader*` (must outlive the
// map), stored as `*const dyn DataLoader` to preserve base-pointer polymorphism.
// `named_data_`/`segments_` are the borrowed flatbuffer vector views, held by
// value (they are fat pointers) with the `'a` lifetime of the underlying
// Program/data buffer, matching the C++ `const Vector*` that must outlive the
// map. The map owns none of these. The deleted move-assignment carries no
// runtime behavior (the type is movable but exposes no assignment-through-
// reference), so its markers collapse onto this struct.
// [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.operator-fn]
// [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.operator-fn]
pub struct PteDataMap<'a> {
    // Data loader, used to load segment data.
    loader_: *const dyn DataLoader,

    // The offset to the first segment in the PTE file, in bytes.
    segment_base_offset_: usize,

    // Named data, containing name and segment index.
    named_data_: FlatbufferNamedData<'a>,

    // Segments, to retrieve offset and size for the loader.
    segments_: FlatbufferDataSegment<'a>,
}

impl<'a> PteDataMap<'a> {
    /// Creates a new DataMap that wraps named_data from the PTE file.
    ///
    /// @param[in] loader The DataLoader that accesses the PTE file.
    /// Note: the loader must outlive the PteDataMap instance.
    /// @param[in] segment_base_offset The offset to the first segment in the PTE
    /// file, in bytes.
    /// @param[in] named_data The named_data from the PTE file. Note: the pointer
    /// passed here must outlive the PteDataMap instance.
    /// @param[in] segments The segments from the PTE file. Note: the pointer
    /// passed here must outlive the PteDataMap instance.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn]
    //
    // PORT-NOTE: the C++ `create` null-checks `loader != nullptr && named_data
    // != nullptr && segments != nullptr`. Here `named_data`/`segments` are
    // non-nullable flatbuffer Vector values (the nullable `const Vector*` is
    // resolved by the caller before reaching this function), so only the
    // `loader` pointer retains a null check.
    pub fn create(
        loader: *const dyn DataLoader,
        segment_base_offset: usize,
        named_data: FlatbufferNamedData<'a>,
        segments: FlatbufferDataSegment<'a>,
    ) -> Result<PteDataMap<'a>> {
        crate::et_check_or_return_error!(
            !loader.is_null(),
            InvalidArgument,
            "PteDataMap loader, named_data or segments is null; most likely the program does not have any named_data segments"
        );
        Ok(PteDataMap::new(
            loader,
            segment_base_offset,
            named_data,
            segments,
        ))
    }

    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn]
    //
    // PORT-NOTE: private constructor. The C++ copy ctor, copy-assign, and
    // move-assign are all `= delete`d; move-construction is defaulted. No
    // assignment path is provided in Rust; the type is constructed once via
    // `create` and thereafter only moved.
    fn new(
        loader: *const dyn DataLoader,
        segment_base_offset: usize,
        named_data: FlatbufferNamedData<'a>,
        segments: FlatbufferDataSegment<'a>,
    ) -> Self {
        PteDataMap {
            loader_: loader,
            segment_base_offset_: segment_base_offset,
            named_data_: named_data,
            segments_: segments,
        }
    }
}

impl NamedDataMap for PteDataMap<'_> {
    /// The PteDataMap currently only handles opaque data that does not contain
    /// tensor-specific metadata.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-tensor-layout-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-tensor-layout-fn]
    fn get_tensor_layout(&self, _key: &str) -> Result<TensorLayout> {
        Err(Error::NotImplemented)
    }

    /// Retrieve read-only data for the specified key.
    ///
    /// @param[in] key The name of the blob to get data on.
    ///
    /// @return error if the key is not present or data cannot be loaded.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn]
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn]
    fn get_data(&self, key: &str) -> Result<FreeableBuffer> {
        let mut i: u32 = 0;
        while (i as usize) < self.named_data_.len() {
            let named_data_item = self.named_data_.get(i as usize);
            // PORT-NOTE: C++ tests `named_data_item != nullptr &&
            // named_data_item->key() != nullptr`. The Rust flatbuffer `.get()`
            // returns a non-nullable value, so only the nullable `key()`
            // (`Option<&str>`) needs checking.
            crate::et_check_or_return_error!(
                named_data_item.key().is_some(),
                InvalidArgument,
                "Searching for key {}: NamedData at index {} is null",
                key,
                i
            );
            let named_data_key = named_data_item.key().unwrap();
            // PORT-NOTE: C++ compares by exact byte length then `memcmp`
            // (length-first byte comparison, embedded NULs significant, no
            // C-string semantics). Rust `&str` equality is a length-first byte
            // comparison, so `named_data_key == key` is the literal equivalent.
            if named_data_key.len() == key.len() && named_data_key.as_bytes() == key.as_bytes() {
                // Get the segment index.
                let segment_index: usize = named_data_item.segment_index() as usize;

                // Get the segment offset and size.
                crate::et_check_or_return_error!(
                    segment_index < self.segments_.len(),
                    InvalidArgument,
                    "Segment index {} for key {} is out of range for segments size {}",
                    segment_index,
                    key,
                    self.segments_.len()
                );
                let segment_offset: usize = self.segments_.get(segment_index).offset() as usize;
                let segment_size: usize = self.segments_.get(segment_index).size() as usize;
                return unsafe {
                    (*self.loader_).load(
                        // offset=
                        self.segment_base_offset_ + segment_offset,
                        segment_size,
                        &SegmentInfo::new(Type::Constant, 0, core::ptr::null()),
                    )
                };
            }
            i += 1;
        }
        Err(Error::NotFound)
    }

    /// The PteDataMap currently does not implement load_into.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.load-data-into-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.load-data-into-fn]
    fn load_data_into(&self, _key: &str, _buffer: *mut core::ffi::c_void, _size: usize) -> Error {
        Error::NotImplemented
    }

    /// @returns The number of keys in the map.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-num-keys-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-num-keys-fn]
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn]
    fn get_num_keys(&self) -> Result<u32> {
        Ok(self.named_data_.len() as u32)
    }

    /// @returns The key at the specified index, error if index out of bounds.
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-key-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-key-fn]
    // [spec:et:def:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn]
    fn get_key(&self, index: u32) -> Result<*const core::ffi::c_char> {
        crate::et_check_or_return_error!(
            (index as usize) < self.named_data_.len(),
            InvalidArgument,
            "Index out of range: named_data size is {}, received index {}",
            self.named_data_.len(),
            index
        );

        let item = self.named_data_.get(index as usize);
        // PORT-NOTE: C++ tests `item != nullptr && item->key() != nullptr`.
        // `.get()` is non-nullable in Rust flatbuffers, so only the nullable
        // `key()` is checked.
        crate::et_check_or_return_error!(
            item.key().is_some(),
            InvalidArgument,
            "NamedData at index {} is null",
            index
        );
        // PORT-NOTE: C++ returns `item->key()->c_str()`, a NUL-terminated
        // pointer into the flatbuffer buffer. The Rust flatbuffers `&str` from
        // `key()` is NUL-terminated in the underlying buffer (flatbuffer strings
        // always are), so its `.as_ptr()` is a valid `const char*` pointing at
        // that same NUL-terminated region, valid for the buffer's lifetime.
        Ok(item.key().unwrap().as_ptr() as *const core::ffi::c_char)
    }
}

// Literal port of runtime/executor/test/pte_data_map_test.cpp.
//
// PORT-NOTE: the C++ fixture builds an in-memory flatbuffer Program carrying
// only `named_data` + `segments` (not a valid full Program, just enough to
// exercise the PteDataMap), writes sample segment bytes to a `TempFile`, and
// wraps that file in a `FileDataLoader`. `TempFile` (extension/testing_util) is
// not ported, so this uses `std::env::temp_dir()` + a unique file directly.
// These tests need no `.pte` fixture env vars and run unconditionally.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::data_loader::file_data_loader::FileDataLoader;
    use crate::runtime::core::result::ResultExt;
    use crate::schema::generated::executorch_flatbuffer;
    use flatbuffers::FlatBufferBuilder;
    use std::io::Write;

    const K_SEGMENT_ALIGNMENT: usize = 16;
    const K_SEGMENT_SIZES: [i32; 2] = [17, 8];
    const K_SEGMENT_OFFSETS: [i32; 2] = [0, (K_SEGMENT_ALIGNMENT * 2) as i32];

    // Owns the built flatbuffer bytes and the sample data file for the duration
    // of a test. `program_bytes` backs the `named_data`/`segments` views handed
    // to `PteDataMap::create`; `loader` reads the sample data file.
    struct Fixture {
        program_bytes: Vec<u8>,
        sample_data: [u8; 64],
        loader: FileDataLoader,
        _temp_path: std::ffi::CString,
        temp_file: std::path::PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.temp_file);
        }
    }

    fn build_program_bytes() -> Vec<u8> {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        // Named data. Note: key2 points to the same segment as key0, and
        // key_invalid points at segment_index=10 (out of range).
        let named_specs: [(&str, u32); 4] =
            [("key0", 0), ("key1", 1), ("key2", 0), ("key_invalid", 10)];
        let mut named_offsets = Vec::new();
        for (key, seg) in named_specs {
            let key_off = builder.create_string(key);
            named_offsets.push(executorch_flatbuffer::NamedData::create(
                &mut builder,
                &executorch_flatbuffer::NamedDataArgs {
                    key: Some(key_off),
                    segment_index: seg,
                },
            ));
        }
        let named_data = builder.create_vector(&named_offsets);

        // Segments.
        let seg0 = executorch_flatbuffer::DataSegment::create(
            &mut builder,
            &executorch_flatbuffer::DataSegmentArgs {
                offset: 0,
                size: K_SEGMENT_SIZES[0] as u64,
            },
        );
        let seg1 = executorch_flatbuffer::DataSegment::create(
            &mut builder,
            &executorch_flatbuffer::DataSegmentArgs {
                offset: (K_SEGMENT_ALIGNMENT * 2) as u64,
                size: K_SEGMENT_SIZES[1] as u64,
            },
        );
        let segments = builder.create_vector(&[seg0, seg1]);

        let program = executorch_flatbuffer::Program::create(
            &mut builder,
            &executorch_flatbuffer::ProgramArgs {
                version: 0,
                segments: Some(segments),
                named_data: Some(named_data),
                ..Default::default()
            },
        );
        builder.finish_minimal(program);
        builder.finished_data().to_vec()
    }

    fn setup() -> Fixture {
        crate::runtime::platform::runtime::runtime_init();

        let program_bytes = build_program_bytes();

        // Create sample segment data.
        let mut sample_data = [0u8; 64];
        for b in sample_data.iter_mut().take(K_SEGMENT_SIZES[0] as usize) {
            *b = 1;
        }
        for i in (K_SEGMENT_OFFSETS[1] as usize)
            ..(K_SEGMENT_OFFSETS[1] as usize + K_SEGMENT_SIZES[1] as usize)
        {
            sample_data[i] = 2;
        }

        // Write it to a unique temp file (TempFile analog).
        let temp_file = std::env::temp_dir().join(format!(
            "et_pte_data_map_{}_{:p}.bin",
            std::process::id(),
            &sample_data as *const _
        ));
        {
            let mut f = std::fs::File::create(&temp_file).expect("create temp file");
            f.write_all(&sample_data).expect("write sample data");
        }
        let temp_path = std::ffi::CString::new(temp_file.to_str().unwrap()).unwrap();

        let loader = FileDataLoader::from(temp_path.as_ptr(), K_SEGMENT_ALIGNMENT);
        assert_eq!(ResultExt::error(&loader), Error::Ok);
        let loader = r_into(loader);

        Fixture {
            program_bytes,
            sample_data,
            loader,
            _temp_path: temp_path,
            temp_file,
        }
    }

    // Moves the Ok value out of a Result without touching the (identical) error
    // path — the tests assert Ok first.
    fn r_into<T>(r: Result<T>) -> T {
        r.unwrap_or_else(|_| panic!("expected Ok result"))
    }

    // Rebuilds the flatbuffer Program view over the owned bytes and returns its
    // (named_data, segments) vectors, mirroring `program_->named_data()` /
    // `program_->segments()` in the C++ fixture.
    fn program_views(bytes: &[u8]) -> (FlatbufferNamedData<'_>, FlatbufferDataSegment<'_>) {
        let program = unsafe { executorch_flatbuffer::root_as_program_unchecked(bytes) };
        (program.named_data().unwrap(), program.segments().unwrap())
    }

    fn make_data_map<'a>(fx: &'a Fixture) -> PteDataMap<'a> {
        let (named_data, segments) = program_views(&fx.program_bytes);
        let dm = PteDataMap::create(
            &fx.loader as *const FileDataLoader as *const dyn DataLoader,
            0,
            named_data,
            segments,
        );
        assert!(ResultExt::ok(&dm));
        r_into(dm)
    }

    fn cstr_str(p: *const core::ffi::c_char) -> &'static str {
        unsafe { core::ffi::CStr::from_ptr(p).to_str().unwrap() }
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn/test]
    // also verifies the private constructor reached through `create`.
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.pte-data-map-fn/test]
    #[test]
    fn pte_data_map_test_load() {
        let fx = setup();
        let (named_data, segments) = program_views(&fx.program_bytes);
        let dm = PteDataMap::create(
            &fx.loader as *const FileDataLoader as *const dyn DataLoader,
            0,
            named_data,
            segments,
        );
        assert!(ResultExt::ok(&dm));
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.create-fn/test]
    #[test]
    fn pte_data_map_test_load_fail() {
        let fx = setup();
        let (named_data, segments) = program_views(&fx.program_bytes);
        // A null `*const dyn DataLoader` (fat-pointer null formed from a null
        // concrete pointer), matching the C++ `nullptr` loader.
        let null_loader: *const dyn DataLoader =
            core::ptr::null::<FileDataLoader>() as *const dyn DataLoader;
        let dm = PteDataMap::create(null_loader, 0, named_data, segments);
        assert_eq!(ResultExt::error(&dm), Error::InvalidArgument);
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-tensor-layout-fn/test]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.load-data-into-fn/test]
    #[test]
    fn pte_data_map_test_unimplemented_methods() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let result = dm.get_tensor_layout("sample_key");
        assert_eq!(ResultExt::error(&result), Error::NotImplemented);

        let err = dm.load_data_into("sample_key", core::ptr::null_mut(), 0);
        assert_eq!(err, Error::NotImplemented);
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-num-keys-fn/test]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-uint32-t-pte-data-map.get-num-keys-fn/test]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-key-fn/test]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-const-char-pte-data-map.get-key-fn/test]
    #[test]
    fn pte_data_map_test_keys() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let num_keys = dm.get_num_keys();
        assert_eq!(ResultExt::error(&num_keys), Error::Ok);
        assert_eq!(*ResultExt::get(&num_keys), 4);

        let key0 = dm.get_key(0);
        assert_eq!(cstr_str(*ResultExt::get(&key0)), "key0");
        let key1 = dm.get_key(1);
        assert_eq!(cstr_str(*ResultExt::get(&key1)), "key1");
        let key2 = dm.get_key(2);
        assert_eq!(cstr_str(*ResultExt::get(&key2)), "key2");

        let key_invalid = dm.get_key(3);
        assert_eq!(cstr_str(*ResultExt::get(&key_invalid)), "key_invalid");

        let nonexistent_key = dm.get_key(10);
        assert_eq!(ResultExt::error(&nonexistent_key), Error::InvalidArgument);
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn/test]
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.result-freeable-buffer-pte-data-map.get-data-fn/test]
    #[test]
    fn pte_data_map_test_get_data() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let mut data0 = dm.get_data("key0");
        assert_eq!(ResultExt::error(&data0), Error::Ok);
        assert_eq!(ResultExt::get(&data0).size(), K_SEGMENT_SIZES[0] as usize);
        assert_eq!(
            fb_slice(&data0),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );

        let mut data1 = dm.get_data("key1");
        assert_eq!(ResultExt::error(&data1), Error::Ok);
        assert_eq!(ResultExt::get(&data1).size(), K_SEGMENT_SIZES[1] as usize);
        assert_eq!(
            fb_slice(&data1),
            &fx.sample_data[K_SEGMENT_OFFSETS[1] as usize
                ..K_SEGMENT_OFFSETS[1] as usize + K_SEGMENT_SIZES[1] as usize]
        );

        let mut data2 = dm.get_data("key2");
        assert_eq!(ResultExt::error(&data2), Error::Ok);
        // key0 and key2 point to the same segment.
        assert_eq!(ResultExt::get(&data2).size(), K_SEGMENT_SIZES[0] as usize);
        assert_eq!(
            fb_slice(&data2),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );

        ResultExt::get_mut(&mut data0).free();
        ResultExt::get_mut(&mut data1).free();
        ResultExt::get_mut(&mut data2).free();

        // key_invalid contains segment_index=10, out of range for 2 segments.
        let data_invalid = dm.get_data("key_invalid");
        assert_eq!(ResultExt::error(&data_invalid), Error::InvalidArgument);

        let data_nonexistent = dm.get_data("nonexistent_key");
        assert_eq!(ResultExt::error(&data_nonexistent), Error::NotFound);
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn/test]
    #[test]
    fn pte_data_map_test_free_and_reload() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let mut data0 = dm.get_data("key0");
        assert_eq!(ResultExt::error(&data0), Error::Ok);
        assert_eq!(ResultExt::get(&data0).size(), K_SEGMENT_SIZES[0] as usize);
        assert_eq!(
            fb_slice(&data0),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );
        ResultExt::get_mut(&mut data0).free();

        let mut data0_reload = dm.get_data("key0");
        assert_eq!(ResultExt::error(&data0_reload), Error::Ok);
        assert_eq!(
            ResultExt::get(&data0_reload).size(),
            K_SEGMENT_SIZES[0] as usize
        );
        assert_eq!(
            fb_slice(&data0_reload),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );
        ResultExt::get_mut(&mut data0_reload).free();
    }

    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.get-data-fn/test]
    #[test]
    fn pte_data_map_test_reload_and_free() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let mut data0 = dm.get_data("key0");
        assert_eq!(ResultExt::error(&data0), Error::Ok);
        assert_eq!(ResultExt::get(&data0).size(), K_SEGMENT_SIZES[0] as usize);
        assert_eq!(
            fb_slice(&data0),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );

        let mut data0_reload = dm.get_data("key0");
        assert_eq!(ResultExt::error(&data0_reload), Error::Ok);
        assert_eq!(
            ResultExt::get(&data0_reload).size(),
            K_SEGMENT_SIZES[0] as usize
        );
        assert_eq!(
            fb_slice(&data0_reload),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );

        // Free data0; data0_reload must still be valid (the FileDataLoader owns a
        // separate mapping/copy).
        ResultExt::get_mut(&mut data0).free();
        assert_eq!(
            ResultExt::get(&data0_reload).size(),
            K_SEGMENT_SIZES[0] as usize
        );
        assert_eq!(
            fb_slice(&data0_reload),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );

        ResultExt::get_mut(&mut data0_reload).free();
    }

    // Views the loaded FreeableBuffer contents as a byte slice.
    fn fb_slice(r: &Result<FreeableBuffer>) -> &[u8] {
        let fb = ResultExt::get(r);
        unsafe { core::slice::from_raw_parts(fb.data() as *const u8, fb.size()) }
    }

    // C++ deletes the copy ctor, copy-assign, and move-assign but keeps move
    // construction (PteDataMap travels through Result<PteDataMap>). The Rust
    // analog is a move-only value: no Clone and no assignment-through-reference,
    // so the only transfer is a move that leaves the source binding statically
    // unusable — no duplication of the borrowed loader/named_data/segments
    // state. Moving the map must carry that state intact: the moved-to binding
    // answers get_num_keys/get_key/get_data identically.
    // [spec:et:sem:pte-data-map.executorch.et-runtime-namespace.internal.pte-data-map.operator-fn/test]
    #[test]
    fn pte_data_map_test_move_preserves_state() {
        let fx = setup();
        let dm = make_data_map(&fx);

        let num_keys_before = *ResultExt::get(&dm.get_num_keys());
        assert_eq!(num_keys_before, 4);

        // `dm` is moved; any further use of it is a compile error (the deleted
        // assignment/copy operators' contract).
        let moved = dm;

        let num_keys = moved.get_num_keys();
        assert_eq!(ResultExt::error(&num_keys), Error::Ok);
        assert_eq!(*ResultExt::get(&num_keys), num_keys_before);

        let key0 = moved.get_key(0);
        assert_eq!(cstr_str(*ResultExt::get(&key0)), "key0");

        let mut data0 = moved.get_data("key0");
        assert_eq!(ResultExt::error(&data0), Error::Ok);
        assert_eq!(ResultExt::get(&data0).size(), K_SEGMENT_SIZES[0] as usize);
        assert_eq!(
            fb_slice(&data0),
            &fx.sample_data[..K_SEGMENT_SIZES[0] as usize]
        );
        ResultExt::get_mut(&mut data0).free();
    }
}
