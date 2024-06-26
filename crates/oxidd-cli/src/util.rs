use std::fmt;
use std::time::Duration;

use oxidd::util::AllocResult;

// spell-checker:ignore subsec

/// Human-readable durations
pub struct HDuration(pub Duration);

impl fmt::Display for HDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let d = self.0;
        let s = d.as_secs();
        if s >= 60 {
            let (m, s) = (s / 60, s % 60);
            let (h, m) = (m / 60, m % 60);
            if h == 0 {
                return write!(f, "{m} m {s} s");
            }
            let (d, h) = (h / 60, h % 60);
            if d == 0 {
                return write!(f, "{h} h {m} m {s} s");
            }
            return write!(f, "{d} d {h} h {m} m {s} s");
        }
        if s != 0 {
            return write!(f, "{:.3} s", d.as_secs_f32());
        }
        let ms = d.subsec_millis();
        if ms != 0 {
            return write!(f, "{ms} ms");
        }
        let us = d.subsec_micros();
        if us != 0 {
            return write!(f, "{us} us");
        }
        write!(f, "{} ns", d.subsec_nanos())
    }
}

pub(crate) fn handle_oom_fn<T>(value: AllocResult<T>, file: &str, line: u32) -> T {
    match value {
        Ok(v) => v,
        Err(_) => {
            eprintln!("Out of memory ({file}:{line})");
            std::process::exit(1);
        }
    }
}

macro_rules! handle_oom {
    ($v:expr) => {
        crate::util::handle_oom_fn($v, file!(), line!())
    };
}
pub(crate) use handle_oom;
