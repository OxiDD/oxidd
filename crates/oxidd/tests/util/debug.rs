use std::fmt::{Debug, Display, Formatter, Result, Write};

pub type ExplicitBFunc = u32;

pub struct TruthTable<'a, S: AsRef<str>> {
    pub(crate) vars: u32,
    pub(crate) columns: &'a [(S, ExplicitBFunc)],
}

impl<S: AsRef<str>> Display for TruthTable<'_, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut table_width = 0;
        for var in 0..self.vars {
            write!(f, " x{var} |")?;
            table_width += 5 + if var == 0 { 0 } else { var.ilog10() };
        }
        for (column, _) in self.columns {
            let column = column.as_ref();
            write!(f, "| {column} ")?;
            table_width += 3 + column.len() as u32;
        }
        writeln!(f)?;
        for _ in 0..table_width {
            f.write_char('-')?;
        }
        writeln!(f)?;
        for assignment in 0..1u32 << self.vars {
            for var in 0..self.vars {
                let width = 2 + if var == 0 { 0 } else { var.ilog10() } as usize;
                let val = (assignment >> var) & 1;
                write!(f, " {val:>width$} |")?;
            }
            for (name, func) in self.columns {
                let width = name.as_ref().len();
                let val = (func >> assignment) & 1;
                write!(f, "| {val:>width$} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
impl<S: AsRef<str>> Debug for TruthTable<'_, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        Display::fmt(self, f)
    }
}

pub struct VarSet {
    vars: u32,
    set: u32,
}

impl Debug for VarSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut list = f.debug_list();
        for i in 0..self.vars {
            if (self.set >> i) & 1 != 0 {
                list.entry(&format_args!("x{i}"));
            }
        }
        list.finish()
    }
}

pub struct SetList<'a> {
    pub(crate) vars: u32,
    pub(crate) sets: &'a [u32],
}

impl Debug for SetList<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut map = f.debug_map();
        let vars = self.vars;
        if self.sets.len() == 1 {
            let set = self.sets[0];
            map.entry(&format_args!("v"), &VarSet { vars, set });
        } else {
            for (i, &set) in self.sets.iter().enumerate() {
                map.entry(&format_args!("v{i}"), &VarSet { vars, set });
            }
        }
        map.finish()
    }
}
