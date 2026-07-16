//! Literal port of kernels/portable/cpu/util/broadcast_indexes_range.h.
//!
//! PORT-NOTE: the C++ templates are parameterized on `kNumInputs` and the
//! iterator's `value_type` is `std::array<ssize_t, kNumInputs + 1>`. Stable Rust
//! cannot express `[isize; N + 1]` from a const generic `N` (that needs
//! `generic_const_exprs`), so this port parameterizes the const generic on the
//! TOTAL tensor count `NT = kNumInputs + 1` instead. The `value_type` and the
//! member `tensors_` (C++ `std::array<const Tensor*, kNumInputs + 1>`) are both
//! `[_; NT]`. Call sites use `BroadcastIndexesRange::<3>` where the C++ wrote
//! `BroadcastIndexesRange<2>`.
//!
//! This single Rust module covers both the general C++
//! `BroadcastIndexesIterator<kNumInputs, support_noncontiguous_input_tensors>`
//! and the `<1, false>` specialization: the specialization's behavior (single
//! non-broadcast input, `current_indexes_[1] == current_indexes_[0]`) falls out
//! of the general fast path where `output_dim_or_zero_if_no_broadcasting_ == 0`.

use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index;
use crate::runtime::core::array_ref::ArrayRef;
use crate::runtime::core::exec_aten::util::tensor_util::K_TENSOR_DIMENSION_LIMIT;
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::core::portable_type::tensor_impl::{SizesType, ssize_t};

// PORT-NOTE: `ET_CHECK_MSG` is the C++ fatal check; mirrored with a local
// `runtime_abort` on failure. Format arguments are dropped since a fatal abort
// follows.
macro_rules! et_check_msg {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            crate::runtime::platform::abort::runtime_abort();
        }
    };
}

// [spec:et:def:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn]
// [spec:et:sem:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn]
//
// NOTE: we bake ArrayRef iterators being pointers into the return type here
// because we assume that iterators are portable across ArrayRef copies.
pub fn arrayref_begin_ignoring_leading_1s(arr: ArrayRef<SizesType>) -> *const SizesType {
    let mut p = arr.begin();
    let end = arr.end();
    while p != end && unsafe { *p } == 1 {
        p = unsafe { p.add(1) };
    }
    p
}

// [spec:et:def:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]
// [spec:et:sem:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn]
pub fn sizes_match_ignoring_leading_1s(lhs: ArrayRef<SizesType>, rhs: ArrayRef<SizesType>) -> bool {
    let lhs_begin = arrayref_begin_ignoring_leading_1s(lhs);
    let lhs_end = lhs.end();

    let rhs_begin = arrayref_begin_ignoring_leading_1s(rhs);
    let rhs_end = rhs.end();

    let lhs_len = unsafe { lhs_end.offset_from(lhs_begin) };
    let rhs_len = unsafe { rhs_end.offset_from(rhs_begin) };

    (lhs_len == rhs_len)
        && (0..lhs_len).all(|i| unsafe { *lhs_begin.offset(i) == *rhs_begin.offset(i) })
}

type ShapeType = [usize; K_TENSOR_DIMENSION_LIMIT];

// [spec:et:def:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn]
// [spec:et:sem:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn]
fn effective_input_broadcast_stride(output: &Tensor, t: &Tensor) -> ShapeType {
    let mut result: ShapeType = [0; K_TENSOR_DIMENSION_LIMIT];
    et_check_msg!(
        t.dim() <= output.dim(),
        "input to broadcasting op should have dim at most output dim, but {} > {}!",
        t.dim() as i32,
        output.dim() as i32
    );

    let num_leading_ones = output.dim() - t.dim();
    for idx in 0..num_leading_ones {
        result[idx as usize] = 0;
    }
    let t_sizes = t.sizes();
    let t_strides = t.strides();
    for idx in num_leading_ones..(num_leading_ones + t.dim()) {
        result[idx as usize] = if *t_sizes.at((idx - num_leading_ones) as usize) == 1 {
            0
        } else {
            *t_strides.at((idx - num_leading_ones) as usize) as usize
        };
    }
    result
}

// [spec:et:def:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn]
// [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn]
// [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false]
//
// PORT-NOTE: `NT` is the total tensor count (`kNumInputs + 1`). This single
// const-generic struct also covers the C++ `BroadcastIndexesIterator<1, false>`
// specialization (see module doc); its `<1, false>` members map onto the general
// items below.
#[derive(Clone)]
pub struct BroadcastIndexesIterator<'a, const NT: usize> {
    // The 0th entry is the current linear index into the output, followed by
    // kNumInputs input indexes.
    current_indexes_: [ssize_t; NT],
    delinearized_output_index_: ShapeType,
    output_dim_or_zero_if_no_broadcasting_: ssize_t,
    output_shape_: ArrayRef<SizesType>,
    effective_input_broadcast_strides_: [ShapeType; NT],
    // PORT-NOTE: `output_shape_` is a raw-pointer `ArrayRef` into the output
    // tensor's sizes buffer; this marker ties the iterator's lifetime `'a` to
    // that tensor, mirroring the C++ `ArrayRef<SizesType> output_shape_` member
    // that references the tensor.
    _marker: core::marker::PhantomData<&'a Tensor<'a>>,
}

impl<'a, const NT: usize> BroadcastIndexesIterator<'a, NT> {
    // Non-end (begin) constructor built from the range's `tensors_` array
    // (element 0 = output, elements 1.. = inputs), per `begin()`.
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]
    fn make_begin(
        tensors: &[&'a Tensor<'a>; NT],
        support_noncontiguous_input_tensors: bool,
    ) -> Self {
        let output = tensors[0];

        let mut all_match = true;
        for jj in 1..NT {
            all_match &= sizes_match_ignoring_leading_1s(tensors[jj].sizes(), output.sizes());
        }
        let output_dim_or_zero_if_no_broadcasting_ =
            if !support_noncontiguous_input_tensors && all_match {
                0
            } else {
                output.dim()
            };

        let mut effective_input_broadcast_strides_ = [[0usize; K_TENSOR_DIMENSION_LIMIT]; NT];
        if support_noncontiguous_input_tensors || output_dim_or_zero_if_no_broadcasting_ != 0 {
            for jj in 1..NT {
                effective_input_broadcast_strides_[jj - 1] =
                    effective_input_broadcast_stride(output, tensors[jj]);
            }
        }

        BroadcastIndexesIterator {
            current_indexes_: [0; NT],
            delinearized_output_index_: [0; K_TENSOR_DIMENSION_LIMIT],
            output_dim_or_zero_if_no_broadcasting_,
            output_shape_: output.sizes(),
            effective_input_broadcast_strides_,
            _marker: core::marker::PhantomData,
        }
    }

    // End constructor `(make_end_t, t, args...)`: sets current_indexes_ =
    // {t.numel(), 0}.
    //
    // The C++ `make_end_t` empty tag type (used to disambiguate this end
    // constructor from the begin constructor) collapses into this distinct
    // `make_end` constructor in the port, so its `<1, false>` markers land here.
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t]
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t.make-end-t-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t.make-end-t-fn]
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn]
    fn make_end(tensors: &[&'a Tensor<'a>; NT]) -> Self {
        let t = tensors[0];
        let mut current_indexes_ = [0; NT];
        current_indexes_[0] = t.numel();
        BroadcastIndexesIterator {
            current_indexes_,
            delinearized_output_index_: [0; K_TENSOR_DIMENSION_LIMIT],
            output_dim_or_zero_if_no_broadcasting_: 0,
            output_shape_: ArrayRef::new(),
            effective_input_broadcast_strides_: [[0usize; K_TENSOR_DIMENSION_LIMIT]; NT],
            _marker: core::marker::PhantomData,
        }
    }

    // [spec:et:def:broadcast-indexes-range.torch.executor.internal.output-index-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.output-index-fn]
    // The `<1, false>` `current_index()` accessor (returns `current_indexes_[0]`)
    // is the same read; its markers land on this shared accessor.
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.current-index-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.current-index-fn]
    pub fn output_index(&self) -> ssize_t {
        self.current_indexes_[0]
    }

    fn output_index_mut(&mut self) -> &mut ssize_t {
        &mut self.current_indexes_[0]
    }

    pub fn deref(&self) -> &[ssize_t; NT] {
        &self.current_indexes_
    }

    // pre-increment operator++.
    //
    // The `<1, false>` post-increment `operator++(int)` copies `*this`, applies
    // pre-increment (`add_to_current_index(1)`), and returns the copy; that
    // collapses onto this shared pre-increment in the port.
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.operator-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.operator-fn]
    pub fn increment(&mut self) {
        *self.output_index_mut() += 1;
        if self.output_dim_or_zero_if_no_broadcasting_ == 0 {
            let output_index = self.output_index();
            for jj in 1..NT {
                self.current_indexes_[jj] = output_index;
            }
            return;
        }
        // TODO: add optimization for particular input tensors not being
        // broadcasted?
        let mut ii = self.output_dim_or_zero_if_no_broadcasting_ - 1;
        while ii >= 0 {
            // You might wonder what happens if output_shape_[ii] == 0. In that
            // case, output.numel() would be 0, and thus we would have begin() ==
            // end() and no iteration.
            if self.delinearized_output_index_[ii as usize] as SizesType
                == *self.output_shape_.at(ii as usize) - 1
            {
                let old_delinearized_output_index_item =
                    self.delinearized_output_index_[ii as usize];
                self.delinearized_output_index_[ii as usize] = 0;
                for jj in 1..NT {
                    self.current_indexes_[jj] -= old_delinearized_output_index_item as ssize_t
                        * self.effective_input_broadcast_strides_[jj - 1][ii as usize] as ssize_t;
                }
            } else {
                self.delinearized_output_index_[ii as usize] += 1;
                for jj in 1..NT {
                    self.current_indexes_[jj] +=
                        self.effective_input_broadcast_strides_[jj - 1][ii as usize] as ssize_t;
                }
                break;
            }
            ii -= 1;
        }
    }

    // [spec:et:def:broadcast-indexes-range.torch.executor.internal.operator-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn]
    // The `<1, false>` `add_to_current_index(n)` (`current_indexes_[0] += n;
    // current_indexes_[1] = current_indexes_[0]`) is the no-broadcasting fast
    // path of this general `operator+=`; its markers land here.
    // [spec:et:def:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn]
    pub fn add_assign(&mut self, n: ssize_t) {
        if n <= 3 {
            for _ in 0..n {
                self.increment();
            }
            return;
        }

        *self.output_index_mut() += n;
        if self.output_dim_or_zero_if_no_broadcasting_ == 0 {
            let output_index = self.output_index();
            for jj in 1..NT {
                self.current_indexes_[jj] = output_index;
            }
            return;
        }
        delinearize_index(
            self.output_index() as usize,
            self.output_shape_,
            self.delinearized_output_index_.as_mut_ptr(),
            self.delinearized_output_index_.len(),
        );
        for ii in 1..NT {
            self.current_indexes_[ii] = 0;
            for jj in 0..(self.output_dim_or_zero_if_no_broadcasting_ as usize) {
                self.current_indexes_[ii] += self.delinearized_output_index_[jj] as ssize_t
                    * self.effective_input_broadcast_strides_[ii - 1][jj] as ssize_t;
            }
        }
    }
}

// PORT-NOTE: the C++ range is consumed by `for (const auto indexes :
// BroadcastIndexesRange<N>(...))`, i.e. `for (; it != end; ++it)` dereferencing
// `*it`. This Rust `Iterator` mirrors that: it starts at `begin`, yields the
// current index array, then advances, stopping once the output index reaches
// `end.output_index()` (== output.numel()). When `output.numel() == 0`, begin
// already compares equal to end and nothing is yielded.
pub struct BroadcastIndexesRangeIter<'a, const NT: usize> {
    it: BroadcastIndexesIterator<'a, NT>,
    end_index_: ssize_t,
    started_: bool,
}

impl<'a, const NT: usize> Iterator for BroadcastIndexesRangeIter<'a, NT> {
    type Item = [ssize_t; NT];

    fn next(&mut self) -> Option<Self::Item> {
        if self.started_ {
            self.it.increment();
        } else {
            self.started_ = true;
        }
        if self.it.output_index() >= self.end_index_ {
            return None;
        }
        Some(*self.it.deref())
    }
}

/// Efficient mechanism for looping over the index space for an output tensor and
/// kNumInputs possibly-broadcasted input tensors.
///
/// PORT-NOTE: `NT` = `kNumInputs + 1` (total tensors); see module doc.
// [spec:et:def:broadcast-indexes-range.broadcast-indexes-range]
pub struct BroadcastIndexesRange<'a, const NT: usize> {
    // tensors_[0] = output; tensors_[1..] = inputs. Mirrors the C++
    // `std::array<const Tensor*, kNumInputs + 1> tensors_`.
    tensors_: [&'a Tensor<'a>; NT],
    support_noncontiguous_input_tensors_: bool,
}

impl<'a, const NT: usize> BroadcastIndexesRange<'a, NT> {
    // [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn]
    //
    // PORT-NOTE: `inputs` provides the `kNumInputs` input tensors; `tensors_[0]`
    // is set to `output` and `tensors_[1..]` to the inputs. `inputs.len()` must
    // equal `NT - 1`.
    pub fn new(output: &'a Tensor<'a>, inputs: &[&'a Tensor<'a>]) -> Self {
        Self::new_with_support(output, inputs, false)
    }

    pub fn new_with_support(
        output: &'a Tensor<'a>,
        inputs: &[&'a Tensor<'a>],
        support_noncontiguous_input_tensors: bool,
    ) -> Self {
        debug_assert_eq!(inputs.len(), NT - 1);
        let mut tensors_: [&'a Tensor<'a>; NT] = [output; NT];
        for jj in 1..NT {
            tensors_[jj] = inputs[jj - 1];
        }
        BroadcastIndexesRange {
            tensors_,
            support_noncontiguous_input_tensors_: support_noncontiguous_input_tensors,
        }
    }

    // [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.begin-fn]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.begin-fn]
    pub fn begin(&self) -> BroadcastIndexesIterator<'a, NT> {
        BroadcastIndexesIterator::make_begin(
            &self.tensors_,
            self.support_noncontiguous_input_tensors_,
        )
    }

    // [spec:et:def:broadcast-indexes-range.broadcast-indexes-range.end-fn]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.end-fn]
    pub fn end(&self) -> BroadcastIndexesIterator<'a, NT> {
        BroadcastIndexesIterator::make_end(&self.tensors_)
    }
}

impl<'a, const NT: usize> IntoIterator for BroadcastIndexesRange<'a, NT> {
    type Item = [ssize_t; NT];
    type IntoIter = BroadcastIndexesRangeIter<'a, NT>;

    fn into_iter(self) -> Self::IntoIter {
        let end_index_ = self.end().output_index();
        let it = self.begin();
        BroadcastIndexesRangeIter {
            it,
            end_index_,
            started_: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::portable::cpu::util::broadcast_util::linearize_access_indexes_tensor;
    use crate::kernels::portable::cpu::util::delinearize_index::delinearize_index_tensor;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::TensorFactory;

    // Mirrors the C++ `range_to_vec`: collect all yielded index arrays by walking
    // begin() -> end() with pre-increment, matching `for (; it != end; ++it)`.
    fn range_to_vec<const NT: usize>(rng: &BroadcastIndexesRange<'_, NT>) -> Vec<[ssize_t; NT]> {
        let end_index = rng.end().output_index();
        let mut it = rng.begin();
        let mut out = Vec::new();
        while it.output_index() < end_index {
            out.push(*it.deref());
            it.increment();
        }
        out
    }

    // Mirrors `*(range.begin() + idx)`: a fresh begin iterator advanced by idx.
    fn begin_plus<const NT: usize>(
        rng: &BroadcastIndexesRange<'_, NT>,
        idx: ssize_t,
    ) -> [ssize_t; NT] {
        let mut it = rng.begin();
        it.add_assign(idx);
        *it.deref()
    }

    // Mirrors the C++ `test_operator_plus` helper.
    fn test_operator_plus<const NT: usize>(range: &BroadcastIndexesRange<'_, NT>) {
        let mut idx: ssize_t = 0;
        for indexes in range_to_vec(range) {
            assert_eq!(begin_plus(range, idx), indexes);
            idx += 1;
        }
    }

    // also exercises the range constructor, the iterator begin/end constructors,
    // the make_end_t tag ctor, and the output_index/current_index accessor.
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.begin-fn/test]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.end-fn/test]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.broadcast-indexes-range-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.broadcast-indexes-iterator-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.make-end-t.make-end-t-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.current-index-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.output-index-fn/test]
    #[test]
    fn broadcast_indexes_range_test_empty() {
        let tf = TensorFactory::<i32>::new();

        let a = tf.make_default(vec![0], vec![]);
        assert_eq!(a.numel(), 0);
        let mut loop_entered = false;
        for _ in BroadcastIndexesRange::<2>::new(&a, &[&a]) {
            loop_entered = true;
        }
        assert!(!loop_entered);
    }

    // [W] -> [W]
    // [spec:et:sem:broadcast-indexes-range.broadcast-indexes-range.begin-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.operator-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn/test]
    #[test]
    fn broadcast_indexes_range_test_one_d_not_broadcasted() {
        let tf = TensorFactory::<i32>::new();

        let out = tf.zeros_default(vec![5]);
        let mut idx: ssize_t = 0;
        let range = BroadcastIndexesRange::<2>::new(&out, &[&out]);
        for elem in range_to_vec(&range) {
            assert_eq!(begin_plus(&range, idx), elem);
            assert_eq!(elem[0], idx);
            idx += 1;
            assert_eq!(elem[0], elem[1]);
        }
    }

    // [1] -> [H, W]
    // [W] -> [H, W]
    // [1, 1] -> [H, W]
    // [1, W] -> [H, W]
    // [H, 1] -> [H, W]
    // [H, W] -> [H, W]
    // Inputs with leading-1 dims ([1], [1,1]) and genuine broadcasting ([4], [3,1])
    // against a [3,4] output make make_begin take the broadcasting path, exercising
    // sizes_match_ignoring_leading_1s, arrayref_begin_ignoring_leading_1s and
    // effective_input_broadcast_stride; test_operator_plus exercises add_to_current_index.
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.sizes-match-ignoring-leading-1s-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.arrayref-begin-ignoring-leading-1s-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.effective-input-broadcast-stride-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.broadcast-indexes-iterator-1-false.add-to-current-index-fn/test]
    #[test]
    fn broadcast_indexes_range_test_one_and_two_d_exhaustive() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![3, 4]);
        let in_0d_scalar = tf.zeros_default(vec![]);
        let in_1d_scalar = tf.zeros_default(vec![1]);
        let in_2d_scalar = tf.zeros_default(vec![1, 1]);

        let in_row = tf.zeros_default(vec![4]);
        let in_col = tf.zeros_default(vec![3, 1]);

        let in_not_broadcast = tf.zeros_default(vec![3, 4]);

        let range = BroadcastIndexesRange::<7>::new(
            &out,
            &[
                &in_0d_scalar,
                &in_1d_scalar,
                &in_2d_scalar,
                &in_row,
                &in_col,
                &in_not_broadcast,
            ],
        );
        let actual = range_to_vec(&range);
        let expected: Vec<[ssize_t; 7]> = vec![
            [0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1],
            [2, 0, 0, 0, 2, 0, 2],
            [3, 0, 0, 0, 3, 0, 3],
            [4, 0, 0, 0, 0, 1, 4],
            [5, 0, 0, 0, 1, 1, 5],
            [6, 0, 0, 0, 2, 1, 6],
            [7, 0, 0, 0, 3, 1, 7],
            [8, 0, 0, 0, 0, 2, 8],
            [9, 0, 0, 0, 1, 2, 9],
            [10, 0, 0, 0, 2, 2, 10],
            [11, 0, 0, 0, 3, 2, 11],
        ];
        assert_eq!(expected, actual);

        test_operator_plus(&range);
    }

    // Make sure nothing is thrown off by a size-1 dim in the output.
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn/test]
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.operator-fn/test]
    #[test]
    fn broadcast_indexes_range_test_one_and_two_d_with_1_in_output_shape_exhaustive() {
        let tf = TensorFactory::<i32>::new();
        const H: i32 = 2;
        const W: i32 = 3;
        let out_row = tf.zeros_default(vec![1, W]);
        let out_col = tf.zeros_default(vec![H, 1]);
        let in_0d_scalar = tf.zeros_default(vec![]);
        let in_1d_scalar = tf.zeros_default(vec![1]);
        let in_2d_scalar = tf.zeros_default(vec![1, 1]);

        let in_row = tf.zeros_default(vec![W]);
        let in_leading_one_row = tf.zeros_default(vec![1, W]);

        let in_col = tf.zeros_default(vec![H, 1]);

        let mut idx: ssize_t = 0;
        let range_row = BroadcastIndexesRange::<6>::new(
            &out_row,
            &[
                &in_0d_scalar,
                &in_1d_scalar,
                &in_2d_scalar,
                &in_row,
                &in_leading_one_row,
            ],
        );
        for indexes in range_to_vec(&range_row) {
            let out_idx = indexes[0];
            let in_0d_idx = indexes[1];
            let in_1d_idx = indexes[2];
            let in_2d_idx = indexes[3];
            let in_row_idx = indexes[4];
            let in_leading_one_row_idx = indexes[5];
            assert_eq!(out_idx, idx);
            idx += 1;
            assert_eq!(in_0d_idx, 0);
            assert_eq!(in_1d_idx, 0);
            assert_eq!(in_2d_idx, 0);
            assert_eq!(in_row_idx, out_idx);
            assert_eq!(in_leading_one_row_idx, out_idx);
        }

        test_operator_plus(&range_row);

        idx = 0;
        let range_col = BroadcastIndexesRange::<5>::new(
            &out_col,
            &[&in_0d_scalar, &in_1d_scalar, &in_2d_scalar, &in_col],
        );
        for indexes in range_to_vec(&range_col) {
            let out_idx = indexes[0];
            let in_0d_idx = indexes[1];
            let in_1d_idx = indexes[2];
            let in_2d_idx = indexes[3];
            let in_col_idx = indexes[4];
            assert_eq!(out_idx, idx);
            idx += 1;
            assert_eq!(in_0d_idx, 0);
            assert_eq!(in_1d_idx, 0);
            assert_eq!(in_2d_idx, 0);
            assert_eq!(in_col_idx, out_idx);
        }

        test_operator_plus(&range_col);
    }

    // [C, H, W] broadcasting, mutation-tested against delinearize_index /
    // linearize_access_indexes.
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn/test]
    // [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.linearize-access-indexes-fn/test]
    #[test]
    fn broadcast_indexes_range_test_three_d_broadcasting() {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![2, 3, 4]);
        let input_tensors = [
            tf.zeros_default(vec![2, 3, 1]),
            tf.zeros_default(vec![2, 1, 4]),
            tf.zeros_default(vec![1, 3, 4]),
            tf.zeros_default(vec![2, 1, 1]),
            tf.zeros_default(vec![1, 3, 1]),
            tf.zeros_default(vec![1, 1, 4]),
            tf.zeros_default(vec![1, 1, 1]),
            tf.zeros_default(vec![2, 3, 4]),
        ];
        let mut idx: ssize_t = 0;
        let inputs: Vec<&Tensor> = input_tensors.iter().collect();
        let range = BroadcastIndexesRange::<9>::new(&out, &inputs);
        for indexes in range_to_vec(&range) {
            let out_idx = indexes[0];
            assert_eq!(out_idx, idx);
            idx += 1;
            let mut out_indexes = [0usize; K_TENSOR_DIMENSION_LIMIT];
            delinearize_index_tensor(
                out_idx as usize,
                &out,
                out_indexes.as_mut_ptr(),
                K_TENSOR_DIMENSION_LIMIT,
            );
            for tensor_idx in 0..input_tensors.len() {
                assert_eq!(
                    indexes[tensor_idx + 1] as usize,
                    linearize_access_indexes_tensor(
                        ArrayRef::from_raw_parts(out_indexes.as_ptr(), K_TENSOR_DIMENSION_LIMIT),
                        out.dim(),
                        &input_tensors[tensor_idx],
                    )
                );
            }
        }
        test_operator_plus(&range);
    }

    // 4-D generalization, mutation-tested against delinearize/linearize.
    // [spec:et:sem:broadcast-indexes-range.torch.executor.internal.broadcast-indexes-iterator-operator-fn/test]
    // [spec:et:sem:delinearize-index.torch.executor.delinearize-index-fn/test]
    // [spec:et:sem:broadcast-util.torch.executor.linearize-access-indexes-fn/test]
    fn four_d_broadcasting_test(n: i32, c: i32, h: i32, w: i32) {
        let tf = TensorFactory::<i32>::new();
        let out = tf.zeros_default(vec![n, c, h, w]);
        let in_broadcast_cw = tf.zeros_default(vec![n, 1, h, 1]);
        let in_broadcast_nh = tf.zeros_default(vec![1, c, 1, w]);

        let mut idx: ssize_t = 0;
        let range = BroadcastIndexesRange::<3>::new(&out, &[&in_broadcast_cw, &in_broadcast_nh]);
        for indexes in range_to_vec(&range) {
            let out_idx = indexes[0];
            let in_cw_idx = indexes[1];
            let in_nh_idx = indexes[2];
            assert_eq!(out_idx, idx);
            idx += 1;
            let mut out_indexes = [0usize; K_TENSOR_DIMENSION_LIMIT];
            delinearize_index_tensor(
                out_idx as usize,
                &out,
                out_indexes.as_mut_ptr(),
                K_TENSOR_DIMENSION_LIMIT,
            );
            assert_eq!(
                in_cw_idx as usize,
                linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(out_indexes.as_ptr(), K_TENSOR_DIMENSION_LIMIT),
                    out.dim(),
                    &in_broadcast_cw,
                )
            );
            assert_eq!(
                in_nh_idx as usize,
                linearize_access_indexes_tensor(
                    ArrayRef::from_raw_parts(out_indexes.as_ptr(), K_TENSOR_DIMENSION_LIMIT),
                    out.dim(),
                    &in_broadcast_nh,
                )
            );
        }

        test_operator_plus(&range);
    }

    #[test]
    fn broadcast_indexes_range_test_four_d_broadcasting() {
        four_d_broadcasting_test(2, 3, 4, 5);
    }

    #[test]
    fn broadcast_indexes_range_test_four_d_broadcasting_with_one_dims_in_output() {
        four_d_broadcasting_test(2, 3, 1, 5);
        four_d_broadcasting_test(2, 1, 3, 1);
    }
}
