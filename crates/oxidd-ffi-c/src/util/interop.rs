//! Fundamental types for interoperability between C/C++ and Rust

use std::borrow::Cow;
use std::ffi::{c_char, OsStr};
use std::mem::MaybeUninit;
use std::str;

/// A pointer to an array plus its size
///
/// The memory referenced by `ptr` is borrowed, i.e., it must not be
/// deallocated.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct slice<T> {
    /// Pointer to the character array of length `len` if non-null
    ptr: *const T,
    /// Element count of the array
    len: usize,
}

impl<T> From<&[T]> for slice<T> {
    fn from(value: &[T]) -> Self {
        slice {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }
}

/// A borrowed string
///
/// Borrowed means that the receiver must not deallocate the character array
/// after use. It is not necessarily null-terminated.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct str_t {
    /// Pointer to the character array of length `len` if non-null
    ptr: *const c_char,
    /// Byte length of the string
    len: usize,
}

impl str_t {
    pub fn to_str_lossy<'a>(self) -> Cow<'a, str> {
        unsafe { c_char_array_to_str(self.ptr, self.len) }
    }

    pub fn to_string_lossy(self) -> String {
        self.to_str_lossy().into_owned()
    }
}

impl From<&str> for str_t {
    fn from(s: &str) -> Self {
        Self {
            ptr: s.as_ptr().cast(),
            len: s.len(),
        }
    }
}

/// Optional
#[derive(Clone, Copy)]
#[repr(C)]
pub struct opt<T: Copy> {
    /// Whether a value is present
    is_some: bool,
    /// The value. May be uninitialized iff `is_some` is false.
    value: MaybeUninit<T>,
}

impl<T: Copy> From<opt<T>> for Option<T> {
    fn from(value: opt<T>) -> Self {
        if value.is_some {
            Some(unsafe { value.value.assume_init() })
        } else {
            None
        }
    }
}

/// Estimation on the amount of remaining elements in an iterator
#[repr(C)]
pub struct size_hint_t {
    /// Lower bound
    pub lower: usize,
    /// Upper bound. `SIZE_MAX` is interpreted as no bound.
    pub upper: usize,
}

impl From<size_hint_t> for (usize, Option<usize>) {
    fn from(hint: size_hint_t) -> Self {
        let upper = if hint.upper == usize::MAX {
            None
        } else {
            Some(hint.upper)
        };
        (hint.lower, upper)
    }
}

/// Iterator
#[repr(C)]
pub struct iter<T: Copy> {
    /// Function to get the next string
    ///
    /// If `is_some` equals to `false` in the function's return value, this
    /// signals the end of the iteration.
    ///
    /// Must not be `NULL`
    next: extern "C" fn(*mut std::ffi::c_void) -> opt<T>,
    /// Function to get an estimate on remaining element count
    ///
    /// @see  `size_hint_t`
    size_hint: Option<extern "C" fn(*mut std::ffi::c_void) -> size_hint_t>,
    /// Context passed as an argument when calling `next` and `size_hint`
    context: *mut std::ffi::c_void,
}

impl<T: Copy> Iterator for iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        Option::from((self.next)(self.context))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(f) = self.size_hint {
            f(self.context).into()
        } else {
            (0, None)
        }
    }
}

/// Equivalent to [`std::slice::from_raw_parts()`], except that it allows `ptr`
/// to be null
pub unsafe fn slice_from_raw_parts<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    }
}

/// Equivalent to [`std::ffi::CStr::from_ptr()`] followed by
/// [`std::ffi::CStr::to_string_lossy()`], except that it allows `ptr` to be
/// null
pub unsafe fn c_char_to_str<'a>(ptr: *const c_char) -> std::borrow::Cow<'a, str> {
    if ptr.is_null() {
        std::borrow::Cow::Borrowed("")
    } else {
        std::ffi::CStr::from_ptr(ptr).to_string_lossy()
    }
}

/// [`slice_from_raw_parts()`] followed by [`String::from_utf8_lossy()`]
pub unsafe fn c_char_array_to_str<'a>(ptr: *const c_char, len: usize) -> std::borrow::Cow<'a, str> {
    const { assert!(std::mem::size_of::<c_char>() == std::mem::size_of::<u8>()) };
    String::from_utf8_lossy(slice_from_raw_parts(ptr.cast(), len))
}

pub unsafe fn c_char_array_to_os_str<'a>(ptr: *const c_char, len: usize) -> &'a OsStr {
    const { assert!(std::mem::size_of::<c_char>() == std::mem::size_of::<u8>()) };
    OsStr::from_encoded_bytes_unchecked(slice_from_raw_parts(ptr.cast(), len))
}

pub fn to_c_str(str: &str) -> *const c_char {
    const { assert!(std::mem::size_of::<c_char>() == std::mem::size_of::<u8>()) };

    let len = str.len();
    if len == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let copy: *mut c_char = libc::malloc(len + 1).cast();
        if copy.is_null() {
            panic!("allocation failed");
        }
        copy.copy_from_nonoverlapping(str.as_ptr().cast(), len);
        copy.add(len).write(0);
        copy
    }
}

/// Functions implemented in `interop.cpp`
///
/// cbindgen:ignore
#[cfg(feature = "cpp")]
mod cpp {
    extern "C" {
        /// C++ `std::string::assign(char *, size_t)`
        ///
        /// `std::ffi::c_void` should be `std::string`, but this is difficult to
        /// realize
        #[link_name = "oxidd$interop$std$string$assign"]
        pub fn std_string_assign(string: *mut std::ffi::c_void, data: *const u8, len: usize);
    }
}
