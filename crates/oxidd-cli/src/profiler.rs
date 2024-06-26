use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use oxidd::Function;
use parking_lot::Mutex;
use serde::Serialize;

use crate::progress::PROGRESS;

#[derive(Clone, Copy, Serialize, Debug)]
struct ReportRecord {
    start: f32,
    end: f32,
    node_count: usize,
}

pub struct Profiler {
    writer: Option<Mutex<csv::Writer<fs::File>>>,
    start: Instant,
}

impl Profiler {
    pub fn new<P: AsRef<Path>>(csv_path: Option<P>) -> Self {
        let writer = csv_path.map(|path| {
            let path = path.as_ref();
            Mutex::new(match csv::Writer::from_path(path) {
                Ok(w) => w,
                Err(err) => {
                    eprintln!("Could not open '{}': {err}", path.display());
                    std::process::exit(1);
                }
            })
        });
        Self {
            writer,
            start: Instant::now(),
        }
    }

    pub fn start_op(&self) -> Option<Instant> {
        PROGRESS.start_op();
        self.writer.as_ref().map(|_| Instant::now())
    }

    pub fn finish_op<F: Function>(&self, op_start: Option<Instant>, func: &F) {
        PROGRESS.finish_op();
        if let Some(ref writer) = self.writer {
            let op_end = Instant::now();
            let record = ReportRecord {
                start: (op_start.unwrap() - self.start).as_secs_f32(),
                end: (op_end - self.start).as_secs_f32(),
                node_count: func.node_count(),
            };
            if let Err(err) = writer.lock().serialize(record) {
                eprintln!("Failed to write record to report CSV: {err}");
                std::process::exit(1);
            }
        }
    }

    pub fn elapsed_time(&self) -> Duration {
        self.start.elapsed()
    }
}
