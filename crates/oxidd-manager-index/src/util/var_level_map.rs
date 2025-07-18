// TODO: should we move this to oxidd-core?

use oxidd_core::{AtomicLevelNo, AtomicVarNo, LevelNo, VarNo};
use std::sync::atomic::Ordering::Relaxed;

pub struct VarLevelMap {
    to_level: Vec<AtomicLevelNo>,
    to_var: Vec<AtomicVarNo>,
}

impl VarLevelMap {
    pub fn new() -> Self {
        Self {
            to_level: Vec::new(),
            to_var: Vec::new(),
        }
    }

    #[allow(unused)] // currently only used in debug assertions
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.to_level.len()
    }

    pub fn extend(&mut self, additional: VarNo) {
        let start = self.to_level.len() as VarNo;
        let end = start + additional;
        self.to_level.extend((start..end).map(AtomicLevelNo::new));
        self.to_var.extend((start..end).map(AtomicVarNo::new));
        debug_assert_eq!(self.to_level.len(), self.to_var.len());
    }

    #[inline(always)]
    pub fn var_to_level(&self, var: VarNo) -> LevelNo {
        self.to_level[var as usize].load(Relaxed)
    }

    #[inline(always)]
    pub fn level_to_var(&self, level: LevelNo) -> VarNo {
        self.to_var[level as usize].load(Relaxed)
    }

    pub fn swap_levels(&self, l1: LevelNo, l2: LevelNo) {
        if l1 != l2 {
            let v1 = self.level_to_var(l1);
            let v2 = self.level_to_var(l2);
            self.to_var[l1 as usize].store(v2, Relaxed);
            self.to_var[l2 as usize].store(v1, Relaxed);
            self.to_level[v1 as usize].store(l2, Relaxed);
            self.to_level[v2 as usize].store(l1, Relaxed);
        }
    }
}
