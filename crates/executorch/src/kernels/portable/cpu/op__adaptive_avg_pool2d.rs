//! Literal port of kernels/portable/cpu/op__adaptive_avg_pool2d.cpp.

use crate::kernels::portable::cpu::pattern::pattern::{AsF32, FromF32};
use crate::kernels::portable::cpu::util::kernel_ops_util::{
    check_adaptive_avg_pool2d_args, get_adaptive_avg_pool2d_out_target_size, output_size_is_valid,
};
use crate::runtime::core::array_ref::{ArrayRef, IntArrayRef};
use crate::runtime::core::error::Error;
use crate::runtime::core::exec_aten::exec_aten::SizesType;
use crate::runtime::core::exec_aten::util::tensor_util::{
    K_TENSOR_DIMENSION_LIMIT, resize_tensor, tensor_is_default_dim_order,
    tensors_have_same_dim_order2,
};
use crate::runtime::core::portable_type::tensor::Tensor;
use crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext;

// [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn]
// [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn]
#[inline]
fn adaptive_start_index(out_idx: i64, out_size: i64, in_size: i64) -> i64 {
    (((out_idx * in_size) as f32) / (out_size as f32)).floor() as i64
}

// [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn]
// [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn]
#[inline]
fn adaptive_end_index(out_idx: i64, out_size: i64, in_size: i64) -> i64 {
    ((((out_idx + 1) * in_size) as f32) / (out_size as f32)).ceil() as i64
}

// [spec:et:def:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn]
// [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn]
pub fn _adaptive_avg_pool2d_out<'a, 'b>(
    ctx: &mut KernelRuntimeContext,
    in_: &Tensor,
    output_size: IntArrayRef,
    out: &'a Tensor<'b>,
) -> &'a Tensor<'b> {
    crate::et_kernel_check!(
        ctx,
        check_adaptive_avg_pool2d_args(in_, output_size, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        tensors_have_same_dim_order2(in_, out),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(ctx, tensor_is_default_dim_order(in_), InvalidArgument, out);

    let mut output_ndim: usize = 0;
    let mut output_sizes: [SizesType; K_TENSOR_DIMENSION_LIMIT] = [0; K_TENSOR_DIMENSION_LIMIT];
    unsafe {
        get_adaptive_avg_pool2d_out_target_size(
            in_,
            output_size,
            output_sizes.as_mut_ptr(),
            &mut output_ndim,
        );
    }

    crate::et_kernel_check!(
        ctx,
        output_size_is_valid(
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim),
            2
        ),
        InvalidArgument,
        out
    );

    crate::et_kernel_check!(
        ctx,
        resize_tensor(
            out,
            ArrayRef::from_raw_parts(output_sizes.as_ptr(), output_ndim)
        ) == Error::Ok,
        InvalidArgument,
        out
    );

    let in_type = in_.scalar_type();

    let op_name = "_adaptive_avg_pool2d.out";

    crate::et_switch_floathbf16_types_and!(Long, in_type, ctx, op_name, CTYPE, {
        let in_ptr: *const CTYPE = in_.const_data_ptr::<CTYPE>();
        let out_ptr: *mut CTYPE = out.mutable_data_ptr::<CTYPE>();

        let ndim = in_.dim() as usize;
        let in_h: i64 = in_.size(ndim as isize - 2) as i64;
        let in_w: i64 = in_.size(ndim as isize - 1) as i64;
        let out_h: i64 = *output_size.at(0);
        let out_w: i64 = *output_size.at(1);

        let channels: usize = in_.size(ndim as isize - 3) as usize;
        let batch_size: usize = if ndim == 4 { in_.size(0) as usize } else { 1 };

        let in_plane_size: usize = (in_h * in_w) as usize;
        let out_plane_size: usize = (out_h * out_w) as usize;

        for b in 0..batch_size {
            for c in 0..channels {
                let plane_idx = b * channels + c;
                let plane_in: *const CTYPE = unsafe { in_ptr.add(plane_idx * in_plane_size) };
                let plane_out: *mut CTYPE = unsafe { out_ptr.add(plane_idx * out_plane_size) };

                let mut oh: i64 = 0;
                while oh < out_h {
                    let ih0 = adaptive_start_index(oh, out_h, in_h);
                    let ih1 = adaptive_end_index(oh, out_h, in_h);

                    let mut ow: i64 = 0;
                    while ow < out_w {
                        let iw0 = adaptive_start_index(ow, out_w, in_w);
                        let iw1 = adaptive_end_index(ow, out_w, in_w);

                        let mut sum: f32 = 0.0;
                        let mut ih = ih0;
                        while ih < ih1 {
                            let mut iw = iw0;
                            while iw < iw1 {
                                sum +=
                                    unsafe { *plane_in.offset((ih * in_w + iw) as isize) }.as_f32();
                                iw += 1;
                            }
                            ih += 1;
                        }

                        let count = (ih1 - ih0) * (iw1 - iw0);
                        unsafe {
                            *plane_out.offset((oh * out_w + ow) as isize) =
                                CTYPE::from_f32(sum / (count as f32));
                        }
                        ow += 1;
                    }
                    oh += 1;
                }
            }
        }
    });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_tensor_close;
    use crate::runtime::core::exec_aten::testing_util::tensor_factory::{
        FactoryValue, TensorFactory,
    };
    use crate::runtime::core::exec_aten::util::scalar_type_util::CppTypeToScalarType;
    use crate::runtime::core::memory_allocator::{MemoryAllocator, MemoryAllocatorBase};
    use crate::runtime::core::portable_type::{BFloat16, Half};

    fn context() -> KernelRuntimeContext<'static> {
        // Mirrors `OperatorTest::SetUp()`'s `runtime_init()` PAL initialization.
        crate::runtime::platform::platform::pal_init();
        KernelRuntimeContext::new(
            crate::extension::module::module::null_event_tracer(),
            core::ptr::null_mut::<MemoryAllocator>() as *mut dyn MemoryAllocatorBase,
        )
    }

    trait FromF64: Copy {
        fn from_f64(v: f64) -> Self;
    }
    impl FromF64 for f32 {
        fn from_f64(v: f64) -> Self {
            v as f32
        }
    }
    impl FromF64 for f64 {
        fn from_f64(v: f64) -> Self {
            v
        }
    }
    impl FromF64 for Half {
        fn from_f64(v: f64) -> Self {
            Half::from_f32(v as f32)
        }
    }
    impl FromF64 for BFloat16 {
        fn from_f64(v: f64) -> Self {
            BFloat16::from_f32(v as f32)
        }
    }

    fn d<T: FromF64>(vals: &[f64]) -> Vec<T> {
        vals.iter().map(|&v| T::from_f64(v)).collect()
    }

    fn test_4d_non_divisible_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let self_ = tf.make_default(
            vec![1, 2, 7, 7],
            d(&[
                -1.5255959033966064,
                -0.7502318024635315,
                -0.6539809107780457,
                -1.6094849109649658,
                -0.1001671776175499,
                -0.6091889142990112,
                -0.9797722697257996,
                -1.6090962886810303,
                -0.7121446132659912,
                0.30372199416160583,
                -0.777314305305481,
                -0.2514548599720001,
                -0.22227048873901367,
                1.6871134042739868,
                0.22842517495155334,
                0.46763551235198975,
                -0.6969721913337708,
                -1.1607614755630493,
                0.6995424032211304,
                0.1990816295146942,
                0.8656923770904541,
                0.24440394341945648,
                -0.6629116535186768,
                0.8073082566261292,
                1.1016807556152344,
                -0.1759360432624817,
                -2.2455577850341797,
                -1.4464579820632935,
                0.0611552819609642,
                -0.617744505405426,
                -0.7980694770812988,
                -0.13162320852279663,
                1.8793456554412842,
                -0.07213178277015686,
                0.15777060389518738,
                -0.7734549641609192,
                0.1990565061569214,
                0.04570228233933449,
                0.15295691788196564,
                -0.47567880153656006,
                -0.11102072149515152,
                0.2927352488040924,
                -0.1578451544046402,
                -0.028787141665816307,
                2.3571109771728516,
                -1.0373387336730957,
                1.5747981071472168,
                -0.6298472285270691,
                -0.9273917078971863,
                0.5451415777206421,
                0.06628026068210602,
                -0.4370401203632355,
                0.7626007199287415,
                0.4415109157562256,
                1.1651384830474854,
                2.0153918266296387,
                0.1374126821756363,
                0.9386447072029114,
                -0.18600109219551086,
                -0.6446393132209778,
                1.5392457246780396,
                -0.8695875406265259,
                -3.331153631210327,
                -0.7478722333908081,
                -0.025502795353531837,
                -1.023330569267273,
                -0.5961856842041016,
                -1.005530595779419,
                -0.21060630679130554,
                -0.007547527551651001,
                1.6734272241592407,
                0.010342830792069435,
                -0.703956663608551,
                -0.18526577949523926,
                -0.9962347149848938,
                -0.831255316734314,
                -0.4610220193862915,
                -0.5600824356079102,
                0.3955761790275574,
                -0.9822770953178406,
                -0.5064865946769714,
                0.09977540373802185,
                0.8165338039398193,
                -0.9135236144065857,
                1.385144591331482,
                -0.8138463497161865,
                -0.9275765419006348,
                1.1119627952575684,
                0.6155447959899902,
                0.19382202625274658,
                -0.1034424677491188,
                -0.15121671557426453,
                -2.102083683013916,
                -0.6200217604637146,
                -1.4782309532165527,
                -1.133418083190918,
                -0.10103506594896317,
                0.3433537483215332,
            ]),
        );
        let output_size_vec: [i64; 2] = [3, 3];
        let output_size = ArrayRef::from_raw_parts(output_size_vec.as_ptr(), 2);
        let out = tf.zeros_default(vec![1, 2, 3, 3]);
        let out_expected = tf.make_default(
            vec![1, 2, 3, 3],
            d(&[
                -0.5498043894767761,
                -0.4718745946884155,
                0.1431751251220703,
                -0.10741885751485825,
                0.1693905144929886,
                -0.01540564838796854,
                0.031902629882097244,
                0.3963559865951538,
                0.18761992454528809,
                -0.08136305958032608,
                -0.12770779430866241,
                -0.02923762798309326,
                -0.218186154961586,
                -0.6554991006851196,
                -0.38993456959724426,
                -0.4862648546695709,
                -0.42452535033226013,
                -0.020265493541955948,
            ]),
        );
        let mut ctx = context();
        _adaptive_avg_pool2d_out(&mut ctx, &self_, output_size, &out);
        assert_tensor_close!(out, out_expected);
    }

    fn test_global_avg_pool_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let self_ = tf.make_default(
            vec![2, 3, 5, 5],
            d(&[
                -0.07664430886507034,
                0.35988152027130127,
                -0.7820167541503906,
                0.07152773439884186,
                0.6648394465446472,
                -0.2867802679538727,
                1.6205507516860962,
                -1.596708059310913,
                -0.05168094485998154,
                -0.30597570538520813,
                0.2485019415616989,
                -0.2225608378648758,
                0.9131518602371216,
                0.20434117317199707,
                0.573984682559967,
                0.41633129119873047,
                0.2615012228488922,
                0.9311244487762451,
                -0.5144619941711426,
                -1.6516656875610352,
                1.046022891998291,
                0.5221869945526123,
                -0.16680915653705597,
                0.05303340032696724,
                0.5637630224227905,
                2.2565572261810303,
                1.8692600727081299,
                -1.195233941078186,
                0.997922420501709,
                0.45918044447898865,
                2.436397075653076,
                -0.1468081772327423,
                -0.47603103518486023,
                -0.29291248321533203,
                -0.3480893075466156,
                0.3487463593482971,
                0.037066876888275146,
                -0.06768559664487839,
                0.429033100605011,
                -0.8681192398071289,
                -0.27121448516845703,
                0.14162853360176086,
                0.1295178383588791,
                0.68141108751297,
                -0.9583004117012024,
                0.06392408162355423,
                0.6589139103889465,
                0.8194591999053955,
                -0.455424040555954,
                2.2124183177948,
                -0.376986563205719,
                -0.14368382096290588,
                0.648020327091217,
                -2.3256072998046875,
                1.268346905708313,
                -0.24832692742347717,
                0.9577805399894714,
                -1.2890100479125977,
                -1.6483157873153687,
                0.8290363550186157,
                -0.837273120880127,
                -0.5296069979667664,
                1.3544285297393799,
                1.3777583837509155,
                -0.07520250231027603,
                -0.42329928278923035,
                0.4216870367527008,
                -0.25764530897140503,
                -1.5834943056106567,
                1.396010398864746,
                -1.0318880081176758,
                1.1390879154205322,
                0.5124644041061401,
                -0.019845306873321533,
                -1.1215951442718506,
                -0.48906704783439636,
                -0.63362717628479,
                -0.7893384099006653,
                -0.8977330327033997,
                -1.8875724077224731,
                0.18704281747341156,
                -0.18132281303405762,
                -0.09136279672384262,
                -0.7349231839179993,
                1.9858763217926025,
                0.3624759018421173,
                -1.3898112773895264,
                -0.1593458354473114,
                0.32896682620048523,
                0.3323023319244385,
                0.3025069236755371,
                -0.9811871647834778,
                1.1038459539413452,
                -0.01265676412731409,
                -0.685114860534668,
                -0.30938950181007385,
                -0.4963502883911133,
                -0.601576566696167,
                -0.47752076387405396,
                -0.29876479506492615,
                0.3931545317173004,
                0.9183048605918884,
                -0.2603555917739868,
                0.050992149859666824,
                -0.3281603753566742,
                1.621635913848877,
                0.859399139881134,
                0.6995598673820496,
                -1.2667319774627686,
                1.2725406885147095,
                -1.2419294118881226,
                -0.8852627873420715,
                -0.3815121650695801,
                0.5052343606948853,
                0.583146333694458,
                -0.021529169753193855,
                2.2462971210479736,
                -0.30710476636886597,
                -0.7002453207969666,
                -2.627337694168091,
                -0.4709088206291199,
                0.030181938782334328,
                0.15957054495811462,
                0.3998355269432068,
                0.017755037173628807,
                2.3573386669158936,
                -0.7965860962867737,
                0.18540963530540466,
                -0.12634699046611786,
                0.31135934591293335,
                2.4666879177093506,
                -1.0854398012161255,
                1.6575441360473633,
                -0.9762163162231445,
                -0.39030343294143677,
                -1.2372092008590698,
                1.0633769035339355,
                -3.092130422592163,
                -1.5128483772277832,
                0.0035125918220728636,
                1.7882674932479858,
                1.2003782987594604,
                -0.3987923860549927,
                2.6922616958618164,
                0.7647452354431152,
                -0.9586629271507263,
                -1.807788610458374,
                -0.3862932622432709,
                0.9918352961540222,
                0.12247885018587112,
            ]),
        );
        let output_size_vec: [i64; 2] = [1, 1];
        let output_size = ArrayRef::from_raw_parts(output_size_vec.as_ptr(), 2);
        let out = tf.zeros_default(vec![2, 3, 1, 1]);
        let out_expected = tf.make_default(
            vec![2, 3, 1, 1],
            d(&[
                0.11181755363941193,
                0.33846473693847656,
                -0.08028639107942581,
                -0.26054587960243225,
                0.050661202520132065,
                0.11346311867237091,
            ]),
        );
        let mut ctx = context();
        _adaptive_avg_pool2d_out(&mut ctx, &self_, output_size, &out);
        assert_tensor_close!(out, out_expected);
    }

    fn test_3d_dtype<T>()
    where
        T: CppTypeToScalarType + FactoryValue + FromF64,
    {
        let tf = TensorFactory::<T>::new();

        let self_ = tf.make_default(
            vec![3, 6, 6],
            d(&[
                -1.0408011674880981,
                0.9166041612625122,
                -1.3041906356811523,
                -1.1096783876419067,
                -1.2187780141830444,
                1.1675924062728882,
                -1.0573921203613281,
                -0.11883937567472458,
                -0.9078398942947388,
                0.34522223472595215,
                -0.5713335871696472,
                -0.2351086586713791,
                1.0076055526733398,
                -0.7528814673423767,
                -0.22499920427799225,
                -0.4326860308647156,
                -1.5071438550949097,
                -0.4585607945919037,
                -0.8480006456375122,
                0.5266043543815613,
                0.029916182160377502,
                -0.04983803257346153,
                1.0650779008865356,
                0.8860366940498352,
                0.4640183746814728,
                -0.4986324608325958,
                0.12886369228363037,
                2.7630667686462402,
                0.14047646522521973,
                1.1191015243530273,
                0.3152315616607666,
                1.7527765035629272,
                -0.7649639248847961,
                1.8298852443695068,
                -0.27840104699134827,
                -0.2719452679157257,
                -1.2944108247756958,
                -0.02431253157556057,
                -0.23535971343517303,
                -0.7087094783782959,
                1.1566312313079834,
                0.42960160970687866,
                -1.1873589754104614,
                -0.7467759251594543,
                -0.9319809079170227,
                -0.8578645586967468,
                -0.9647331237792969,
                -0.0991455465555191,
                -1.0189824104309082,
                0.31567901372909546,
                -1.6035629510879517,
                1.8493320941925049,
                0.04472726583480835,
                1.5852519273757935,
                -0.5912226438522339,
                1.1312177181243896,
                0.7562121748924255,
                -1.2022933959960938,
                -0.5833470225334167,
                -0.44068679213523865,
                -1.9791470766067505,
                0.7787133455276489,
                -0.7748500108718872,
                -0.13975095748901367,
                1.141386866569519,
                -0.635371744632721,
                -1.4702459573745728,
                -0.21338605880737305,
                -0.8706575632095337,
                1.6159113645553589,
                -0.23564793169498444,
                0.9443864226341248,
                2.113386631011963,
                -0.9754034876823425,
                0.17569366097450256,
                -0.13188815116882324,
                -0.27350300550460815,
                0.3355262577533722,
                0.18854673206806183,
                2.143237352371216,
                0.8527002334594727,
                0.09647636860609055,
                -0.06249098479747772,
                0.8268541097640991,
                0.5598673224449158,
                -0.7775934934616089,
                0.3339212238788605,
                0.17591367661952972,
                2.1108040809631348,
                1.0702455043792725,
                0.019540296867489815,
                1.121282935142517,
                -0.10142341256141663,
                1.2330807447433472,
                -2.526270866394043,
                0.07206021994352341,
                0.537361741065979,
                -0.19295911490917206,
                0.5936785936355591,
                0.720313549041748,
                0.050786226987838745,
                -0.4396914839744568,
                0.5056329369544983,
                -0.5394221544265747,
                -0.6008040904998779,
                -1.1164305210113525,
                0.2577155530452728,
                -0.7225574254989624,
            ]),
        );
        let output_size_vec: [i64; 2] = [2, 2];
        let output_size = ArrayRef::from_raw_parts(output_size_vec.as_ptr(), 2);
        let out = tf.zeros_default(vec![3, 2, 2]);
        let out_expected = tf.make_default(
            vec![3, 2, 2],
            d(&[
                -0.3869704306125641,
                -0.4467194378376007,
                0.12286818772554398,
                0.800384521484375,
                -0.7474517226219177,
                0.2705657184123993,
                -0.3592629134654999,
                0.05162075534462929,
                0.5127062797546387,
                0.4608819782733917,
                0.14920975267887115,
                -0.27455487847328186,
            ]),
        );
        let mut ctx = context();
        _adaptive_avg_pool2d_out(&mut ctx, &self_, output_size, &out);
        assert_tensor_close!(out, out_expected);
    }

    // [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn/test]
    // 4D numeric path also verifies the adaptive-pool arg check and out-target-size helper.
    // [spec:et:sem:kernel-ops-util.torch.executor.check-adaptive-avg-pool2d-args-fn/test]
    // [spec:et:sem:kernel-ops-util.torch.executor.get-adaptive-avg-pool2d-out-target-size-fn/test]
    // Non-divisible 3x3-from-7x7 pooling exercises the varying start/end index math.
    // [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-start-index-fn/test]
    // [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-end-index-fn/test]
    #[test]
    fn op_adaptive_avg_pool2d_out_test_non_divisible_4d() {
        // ET_FORALL_FLOATHBF16_TYPES
        test_4d_non_divisible_dtype::<f32>();
        test_4d_non_divisible_dtype::<f64>();
        test_4d_non_divisible_dtype::<Half>();
        test_4d_non_divisible_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn/test]
    #[test]
    fn op_adaptive_avg_pool2d_out_test_global_avg_pool() {
        test_global_avg_pool_dtype::<f32>();
        test_global_avg_pool_dtype::<f64>();
        test_global_avg_pool_dtype::<Half>();
        test_global_avg_pool_dtype::<BFloat16>();
    }

    // [spec:et:sem:op-adaptive-avg-pool2d.torch.executor.native.adaptive-avg-pool2d-out-fn/test]
    #[test]
    fn op_adaptive_avg_pool2d_out_test_three_dimensional() {
        test_3d_dtype::<f32>();
        test_3d_dtype::<f64>();
        test_3d_dtype::<Half>();
        test_3d_dtype::<BFloat16>();
    }
}
