use oxidd_parser::load_file::load_file;
use oxidd_parser::ParseOptionsBuilder;

fn main() {
    let parse_options = ParseOptionsBuilder::default().build().unwrap();

    for arg in std::env::args().skip(1) {
        println!("\nloading {arg} ...");
        if let Some(problem) = load_file(arg, &parse_options) {
            println!("{problem:#?}")
        }
    }
}
