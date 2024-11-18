pub struct Progress {
    enabled: bool,
    current: u64,
    total: u64,
    percent: u32,
}

impl Progress {
    pub fn new(total: u64) -> Self {
        assert_ne!(total, 0);
        let enabled =
            std::env::var("OXIDD_TESTING_PROGRESS").is_ok_and(|v| !v.is_empty() && v != "0");
        let mut this = Self {
            enabled,
            current: 0,
            total,
            percent: 1, // value != 0 ensures that we print below
        };
        this.print();
        this
    }

    pub fn step(&mut self) {
        self.current += 1;
        self.print()
    }

    pub fn done(&mut self) {
        if self.enabled {
            eprintln!();
        }
        if self.current != self.total {
            eprintln!(
                "WARNING: total number has changed ({} -> {})",
                self.total, self.current,
            );
        }
    }

    pub fn print(&mut self) {
        if !self.enabled {
            return;
        }
        let percent = (self.current * 100 / self.total) as u32;
        if percent != self.percent {
            self.percent = percent;
            eprint!("\r{percent} %");
            std::io::Write::flush(&mut std::io::stderr()).unwrap();
        }
    }
}
