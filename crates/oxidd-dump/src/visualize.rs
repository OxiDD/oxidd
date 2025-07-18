//! Export the diagrams over http to other tools (polling for changes)

use std::io;

use std::io::{BufRead, Write};
use std::net::TcpListener;

use oxidd_core::{function::Function, HasLevel, Manager};

/// Serve the provided decision diagram for visualization
///
/// `dd_name` is the name that is sent to the visualization tool.
///
/// `vars` are edges representing *all* variables in the decision diagram. The
/// order does not matter. `var_names` are the names of these variables
/// (optional). If given, there must be `vars.len()` names in the same order as
/// in `vars`.
///
/// `functions` are edges pointing to the root nodes of functions.
/// `function_names` are the corresponding names (optional). If given, there
/// must be `functions.len()` names in the same order as in `function_names`.
///
/// `port` is the port to provide the data on, which defaults to 4000.
pub fn visualize<'id, F: Function>(
    manager: &F::Manager<'id>,
    dd_name: &str,
    functions: &[&F],
    function_names: Option<&[&str]>,
    port: Option<u16>,
) -> io::Result<()>
where
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    <F::Manager<'id> as Manager>::Terminal: crate::AsciiDisplay,
{
    visualize_all(
        [VisualizationInput {
            dd_name,
            manager,
            functions,
            function_names,
        }],
        port,
    )
}

/// Serve the provided decision diagrams for visualization
///
/// `inputs` is the vector of visualizations to send all at once.
///
/// `port` is the port to provide the data on, which defaults to 4000.
pub fn visualize_all<'a, 'id, F: Function + 'a>(
    inputs: impl IntoIterator<Item = VisualizationInput<'a, 'id, F>>,
    port: Option<u16>,
) -> io::Result<()>
where
    F::Manager<'id>: 'a,
    <F::Manager<'id> as Manager>::InnerNode: HasLevel,
    <F::Manager<'id> as Manager>::Terminal: crate::AsciiDisplay,
{
    let port = port.unwrap_or(4000);

    let mut body_buffer = Vec::with_capacity(2 * 1024 * 1024);
    body_buffer.push(b'[');
    for VisualizationInput {
        dd_name,
        function_names,
        functions,
        manager,
    } in inputs
    {
        write!(body_buffer, "{{\"type\":\"{}\",\"name\":\"", F::REPR_ID)?;
        write!(JsonStrWriter(&mut body_buffer), "{dd_name}")?;
        body_buffer.extend_from_slice(b"\",\"diagram\":\"");
        crate::dddmp::export(
            JsonStrWriter(&mut body_buffer),
            manager,
            true,
            dd_name,
            functions,
            function_names,
            |_| false,
        )?;
        body_buffer.extend_from_slice(b"\"},");
    }
    body_buffer.pop(); // trailing ','
    body_buffer.push(b']');

    let listener = TcpListener::bind(("localhost", port))?;
    println!("Data can be read on http://localhost:{port}");

    for stream in listener.incoming() {
        let mut stream = stream?;

        let req_line = {
            let mut stream = io::BufReader::new(&mut stream);
            let mut buffer = String::new();
            stream.read_line(&mut buffer)?;
            buffer
        };

        // Basic routing: only handle GET /diagrams
        if req_line.starts_with("GET /diagrams ") {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\n\
                Content-Type: application/json\r\n\
                Access-Control-Allow-Origin: *\r\n\
                Content-Length: {}\r\n\
                \r\n",
                body_buffer.len(),
            )?;
            stream.write_all(&body_buffer)?;
            break;
        }

        stream.write_all(b"HTTP/1.1 404 NOT FOUND\r\n\r\nNot Found")?;
    }

    println!("Visualization has been sent!");
    Ok(())
}

/// Input for the visualization
pub struct VisualizationInput<'a, 'id, F: Function> {
    /// The manager that the functions are from
    pub manager: &'a F::Manager<'id>,
    /// The name of the diagram
    pub dd_name: &'a str,
    /// The functions to visualize
    pub functions: &'a [&'a F],
    /// The names of the functions to visualize
    pub function_names: Option<&'a [&'a str]>,
}

struct JsonStrWriter<'a>(&'a mut Vec<u8>);

impl std::io::Write for JsonStrWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for &c in buf {
            match c {
                0x08 => self.0.extend_from_slice(b"\\b"),
                b'\t' => self.0.extend_from_slice(b"\\t"),
                b'\n' => self.0.extend_from_slice(b"\\n"),
                0x0c => self.0.extend_from_slice(b"\\ff"),
                b'\r' => self.0.extend_from_slice(b"\\r"),
                b'\\' => self.0.extend_from_slice(b"\\\\"),
                b'"' => self.0.extend_from_slice(b"\\\""),
                _ if c < 0x20 => write!(self.0, "\\u{c:04x}")?,
                0x7f => self.0.extend_from_slice(b"\\u007f"), // ASCII DEL
                _ => self.0.push(c),
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
