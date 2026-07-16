//! Proc-macro codegen for the ExecuTorch Rust port.
//!
//! `#[et_kernel("aten::op.overload")]` on a kernel function generates its
//! "unboxing" wrapper — the `OpFunction` that unpacks the `EValue` argument
//! stack and calls the typed kernel — and registers it under the given operator
//! name. This replaces the C++ codegen (`functions.yaml` → `RegisterKernels`)
//! for the ops it annotates; the per-argument accessor is derived from the
//! kernel's own Rust parameter types, so it can never drift from the signature.
//!
//! ExecuTorch calling convention: the stack is `[inputs.., outs.., returns..]`,
//! so argument `i` (inputs then outs, in signature order) is `stack[i]`; the
//! trailing return-alias slots are ignored. `outs = N` (default 1) says how many
//! trailing parameters are outputs (only affects documentation today, since outs
//! are unpacked exactly like inputs by position).

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{FnArg, Ident, ItemFn, LitInt, LitStr, Token, parse_macro_input};

struct KernelArgs {
    name: LitStr,
    #[allow(dead_code)]
    outs: usize,
}

impl Parse for KernelArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: LitStr = input.parse()?;
        let mut outs = 1usize;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let key: Ident = input.parse()?;
            if key != "outs" {
                return Err(syn::Error::new(key.span(), "expected `outs`"));
            }
            input.parse::<Token![=]>()?;
            outs = input.parse::<LitInt>()?.base10_parse()?;
        }
        Ok(KernelArgs { name, outs })
    }
}

#[proc_macro_attribute]
pub fn et_kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as KernelArgs);
    let func = parse_macro_input!(item as ItemFn);
    let fn_ident = func.sig.ident.clone();

    // Skip the leading `ctx: &mut KernelRuntimeContext`.
    let params: Vec<&FnArg> = func.sig.inputs.iter().skip(1).collect();

    let r = quote!(crate::kernels::registry);
    let mut locals = Vec::new();
    let mut call_args = Vec::new();

    for (i, param) in params.iter().enumerate() {
        let ty = match param {
            FnArg::Typed(pt) => &*pt.ty,
            FnArg::Receiver(_) => {
                return syn::Error::new_spanned(param, "kernel cannot take self")
                    .to_compile_error()
                    .into();
            }
        };
        // Normalize the type to a whitespace-free string for pattern matching.
        let n: String = quote!(#ty).to_string().split_whitespace().collect();
        let idx = i;
        let ev = quote!(#r::arg(stack, #idx));

        // A leading `&` (e.g. `&Option<Tensor>`) doesn't change which accessor
        // to use, only whether the kernel wants the value by reference. Strip it
        // for matching and remember to bind-a-local + pass `&local` when set.
        let (is_ref, base) = match n.strip_prefix('&') {
            Some(rest) => (true, rest.to_string()),
            None => (false, n.clone()),
        };

        // Optionals first (they wrap another type).
        if let Some(inner) = base
            .strip_prefix("Option<")
            .and_then(|s| s.strip_suffix('>'))
        {
            // Optional tensor is the one case a kernel takes by reference
            // (`&Option<Tensor>`), so bind an owned handle to a local.
            if inner.contains("Tensor") {
                let local = format_ident!("__et_arg{}", i);
                locals.push(quote!(let #local = #r::opt_tensor_owned(#ev);));
                if is_ref {
                    call_args.push(quote!(&#local));
                } else {
                    call_args.push(quote!(#local));
                }
                continue;
            }
            let expr = if inner.contains("ScalarType") {
                quote!(#r::opt_dtype(#ev))
            } else if inner.contains("MemoryFormat") {
                quote!(#r::opt_memory_format(#ev))
            } else if inner.contains("Scalar") {
                quote!(#r::opt_scalar(#ev))
            } else if inner.contains("ArrayRef<i64>") || inner.contains("IntArrayRef") {
                quote!(#r::opt_int_list(#ev))
            } else if inner == "i64" {
                quote!(#r::opt_int(#ev))
            } else {
                return unsupported(param, &n);
            };
            call_args.push(expr);
            continue;
        }

        if base.contains("TensorList") || base.contains("ArrayRef<Tensor") {
            call_args.push(quote!(#ev.to_tensor_list()));
        } else if n.contains("Tensor") {
            call_args.push(quote!(#ev.to_tensor()));
        } else if n.contains("IntArrayRef") || n.contains("ArrayRef<i64>") {
            call_args.push(quote!(#ev.to_int_list()));
        } else if n.contains("ScalarType") {
            // A bare (non-optional) ScalarType is passed as an int enum value.
            call_args.push(quote!(#r::dtype(#ev)));
        } else if n.contains("Scalar") {
            // Kernels take `&Scalar`; `to_scalar` returns by value, so bind a local.
            let local = format_ident!("__et_arg{}", i);
            locals.push(quote!(let #local = #ev.to_scalar();));
            if n.starts_with('&') {
                call_args.push(quote!(&#local));
            } else {
                call_args.push(quote!(#local));
            }
        } else if n == "bool" {
            call_args.push(quote!(#ev.to_bool()));
        } else if n == "f64" {
            call_args.push(quote!(#ev.to_double()));
        } else if n == "i64" {
            call_args.push(quote!(#ev.to_int()));
        } else {
            return unsupported(param, &n);
        }
    }

    let boxed_ident = format_ident!("{}__et_boxed", fn_ident);
    let reg_ident = format_ident!("__ET_REG_{}", fn_ident.to_string().to_uppercase());
    let name_lit = &args.name;

    let expanded = quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        pub fn #boxed_ident(
            ctx: &mut crate::runtime::kernel::kernel_runtime_context::KernelRuntimeContext,
            stack: crate::runtime::core::span::Span<*mut crate::runtime::core::evalue::EValue>,
        ) {
            #(#locals)*
            let _ = #fn_ident(ctx, #(#call_args),*);
        }

        #[linkme::distributed_slice(#r::ET_KERNELS)]
        static #reg_ident: #r::KernelReg = #r::KernelReg {
            name: unsafe {
                ::core::ffi::CStr::from_bytes_with_nul_unchecked(
                    ::core::concat!(#name_lit, "\0").as_bytes(),
                )
            },
            op: #boxed_ident,
        };
    };
    expanded.into()
}

fn unsupported(param: &FnArg, n: &str) -> TokenStream {
    syn::Error::new_spanned(
        param,
        format!("et_kernel: unsupported parameter type `{n}` (add a mapping in executorch-macros)"),
    )
    .to_compile_error()
    .into()
}
