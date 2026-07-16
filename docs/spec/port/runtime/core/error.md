# runtime/core/error.h

> [spec:et:def:error.executorch.runtime.error]
> enum class Error : error_code_t {
>   Ok = 0x00;
>   Internal = 0x01;
>   InvalidState = 0x2;
>   EndOfMethod = 0x03;
>   AlreadyLoaded = 0x04;
>   NotSupported = 0x10;
>   NotImplemented = 0x11;
>   InvalidArgument = 0x12;
>   InvalidType = 0x13;
>   OperatorMissing = 0x14;
>   RegistrationExceedingMaxKernels = 0x15;
>   RegistrationAlreadyRegistered = 0x16;
>   NotFound = 0x20;
>   MemoryAllocationFailed = 0x21;
>   AccessFailed = 0x22;
>   InvalidProgram = 0x23;
>   InvalidExternalData = 0x24;
>   OutOfResources = 0x25;
>   DelegateInvalidCompatibility = 0x30;
>   DelegateMemoryAllocationFailed = 0x31;
>   DelegateInvalidHandle = 0x32;
> }

> [spec:et:def:error.executorch.runtime.error-code-t]
> typedef uint32_t error_code_t

> [spec:et:def:error.executorch.runtime.to-string-fn]
> constexpr const char* to_string(const Error error)

> [spec:et:sem:error.executorch.runtime.to-string-fn]
> Pure, `constexpr`, side-effect-free mapping from an `Error` enum value to a
> stable, statically-allocated, null-terminated C string naming that value.
> The returned pointer refers to a string literal with static storage
> duration; the caller must not free or mutate it, and it remains valid for
> the lifetime of the program.
>
> Behavior is a single `switch` over the argument `error`, returning exactly
> one of these strings (each is the enum name prefixed with `"Error::"`):
> - `Error::Ok` (0x00) -> `"Error::Ok"`
> - `Error::Internal` (0x01) -> `"Error::Internal"`
> - `Error::InvalidState` (0x02) -> `"Error::InvalidState"`
> - `Error::EndOfMethod` (0x03) -> `"Error::EndOfMethod"`
> - `Error::AlreadyLoaded` (0x04) -> `"Error::AlreadyLoaded"`
> - `Error::NotSupported` (0x10) -> `"Error::NotSupported"`
> - `Error::NotImplemented` (0x11) -> `"Error::NotImplemented"`
> - `Error::InvalidArgument` (0x12) -> `"Error::InvalidArgument"`
> - `Error::InvalidType` (0x13) -> `"Error::InvalidType"`
> - `Error::OperatorMissing` (0x14) -> `"Error::OperatorMissing"`
> - `Error::RegistrationExceedingMaxKernels` (0x15) -> `"Error::RegistrationExceedingMaxKernels"`
> - `Error::RegistrationAlreadyRegistered` (0x16) -> `"Error::RegistrationAlreadyRegistered"`
> - `Error::NotFound` (0x20) -> `"Error::NotFound"`
> - `Error::MemoryAllocationFailed` (0x21) -> `"Error::MemoryAllocationFailed"`
> - `Error::AccessFailed` (0x22) -> `"Error::AccessFailed"`
> - `Error::InvalidProgram` (0x23) -> `"Error::InvalidProgram"`
> - `Error::InvalidExternalData` (0x24) -> `"Error::InvalidExternalData"`
> - `Error::OutOfResources` (0x25) -> `"Error::OutOfResources"`
> - `Error::DelegateInvalidCompatibility` (0x30) -> `"Error::DelegateInvalidCompatibility"`
> - `Error::DelegateMemoryAllocationFailed` (0x31) -> `"Error::DelegateMemoryAllocationFailed"`
> - `Error::DelegateInvalidHandle` (0x32) -> `"Error::DelegateInvalidHandle"`
>
> Any value not listed above (i.e. any bit pattern that does not correspond to
> a declared `Error` enumerator) falls through to the `default` case and
> returns `"Error::Unknown"`. The function never returns null and never
> throws.

