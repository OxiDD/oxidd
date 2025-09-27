//! Visualization with [OxiDD-vis](https://oxidd.net/vis)

use std::fmt;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};

use oxidd_core::function::{Function, INodeOfFunc, TermOfFunc};
use oxidd_core::HasLevel;

use crate::AsciiDisplay;

/// [OxiDD-vis]-compatible decision diagram exporter that
/// serves decision diagram dumps on localhost via HTTP
///
/// [OxiDD-vis] is a webapp that runs locally in your browser. You can directly
/// send decision diagrams to it via an HTTP connection. Here, the `Visualizer`
/// acts as a small HTTP server that accepts connections once you call
/// [`Visualizer::serve()`]. The webapp repeatedly polls on the configured port
/// to directly display the decision diagrams then.
///
/// [OxiDD-vis]: https://oxidd.net/vis
pub struct Visualizer {
    port: u16,
    buf: Vec<u8>,
}

impl Default for Visualizer {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Visualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            port: 4000,
            buf: Vec::with_capacity(2 * 1024 * 1024), // 2 MiB
        }
    }

    /// Customize the port on which to serve the visualization data
    ///
    /// The default port is 4000.
    ///
    /// Returns `self`.
    #[inline(always)]
    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    fn add_diagram_start(&mut self, ty: &str, diagram_name: &str) {
        let buf = &mut self.buf;
        if !buf.is_empty() {
            buf.push(b',');
        }

        buf.extend_from_slice(b"{\"type\":\"");
        buf.extend_from_slice(ty.as_bytes());
        buf.extend_from_slice(b"\",\"name\":\"");
        json_escape(buf, diagram_name.as_bytes());
        buf.extend_from_slice(b"\",\"diagram\":\"");
    }

    /// Add a decision diagram for visualization
    ///
    /// `diagram_name` can be used as an identifier in case you add multiple
    /// diagrams. `functions` is an iterator over [`Function`]s.
    ///
    /// The visualization includes all nodes reachable from the root nodes
    /// referenced by `functions`. If you wish to name the functions, use
    /// [`Self::add_with_names()`].
    ///
    /// Returns `self` to allow chaining.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxidd_core::function::Function;
    /// # use oxidd_dump::Visualizer;
    /// # fn vis<'id, F: Function>(manager: &F::Manager<'id>, f0: &F, f1: &F)
    /// # where
    /// #    oxidd_core::function::INodeOfFunc<'id, F>: oxidd_core::HasLevel,
    /// #    oxidd_core::function::TermOfFunc<'id, F>: oxidd_dump::AsciiDisplay,
    /// # {
    /// Visualizer::new()
    ///     .add("my_diagram", manager, [f0, f1])
    ///     .serve()
    ///     .expect("failed to serve diagram due to I/O error")
    /// # }
    /// ```
    pub fn add<'id, FR: std::ops::Deref>(
        mut self,
        diagram_name: &str,
        manager: &<FR::Target as Function>::Manager<'id>,
        functions: impl IntoIterator<Item = FR>,
    ) -> Self
    where
        FR::Target: Function,
        INodeOfFunc<'id, FR::Target>: HasLevel,
        TermOfFunc<'id, FR::Target>: AsciiDisplay,
    {
        self.add_diagram_start(FR::Target::REPR_ID, diagram_name);
        crate::dddmp::ExportSettings::default()
            .strict(false) // adjust names without error
            .ascii()
            .diagram_name(diagram_name)
            .export(JsonStrWriter(&mut self.buf), manager, functions)
            .unwrap(); // writing to a Vec<u8> should not lead to I/O errors
        self.buf.extend_from_slice(b"\"}");

        self
    }

    /// Add a decision diagram for visualization
    ///
    /// `diagram_name` can be used as an identifier in case you add multiple
    /// diagrams. `functions` is an iterator over pairs of a [`Function`] and
    /// a name.
    ///
    /// The visualization includes all nodes reachable from the root nodes
    /// referenced by `functions`.
    ///
    /// Returns `self` to allow chaining.
    ///
    /// # Example
    ///
    /// ```
    /// # use oxidd_core::function::Function;
    /// # use oxidd_dump::Visualizer;
    /// # fn vis<'id, F: Function>(manager: &F::Manager<'id>, phi: &F, res: &F)
    /// # where
    /// #    oxidd_core::function::INodeOfFunc<'id, F>: oxidd_core::HasLevel,
    /// #    oxidd_core::function::TermOfFunc<'id, F>: oxidd_dump::AsciiDisplay,
    /// # {
    /// Visualizer::new()
    ///     .add_with_names("my_diagram", manager, [(phi, "ϕ"), (res, "result")])
    ///     .serve()
    ///     .expect("failed to serve diagram due to I/O error")
    /// # }
    /// ```
    pub fn add_with_names<'id, FR: std::ops::Deref, D: fmt::Display>(
        mut self,
        diagram_name: &str,
        manager: &<FR::Target as Function>::Manager<'id>,
        functions: impl IntoIterator<Item = (FR, D)>,
    ) -> Self
    where
        FR::Target: Function,
        INodeOfFunc<'id, FR::Target>: HasLevel,
        TermOfFunc<'id, FR::Target>: AsciiDisplay,
    {
        self.add_diagram_start(FR::Target::REPR_ID, diagram_name);
        crate::dddmp::ExportSettings::default()
            .strict(false) // adjust names without error
            .ascii()
            .diagram_name(diagram_name)
            .export_with_names(JsonStrWriter(&mut self.buf), manager, functions)
            .unwrap(); // writing to a Vec<u8> should not lead to I/O errors
        self.buf.extend_from_slice(b"\"}");

        self
    }

    /// Remove all previously added decision diagrams
    #[inline(always)]
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Serve the provided decision diagram for visualization
    ///
    /// Blocks until the visualization has been fetched by
    /// [OxiDD-vis](https://oxidd.net/vis) (or another compatible tool).
    ///
    /// On success, all previously added decision diagrams are removed from the
    /// internal buffer. On error, the internal buffer is left as-is.
    pub fn serve(&mut self) -> io::Result<()> {
        let port = self.port;
        let listener = TcpListener::bind(("localhost", port))?;
        println!("Data can be read on http://localhost:{port}");

        let mut buf = EMPTY_RECV_BUF;
        while !self.handle_client(listener.accept()?.0, &mut buf)? {}
        Ok(())
    }

    /// Non-blocking version of [`Self::serve()`]
    ///
    /// Unlike [`Self::serve()`], this method sets the [`TcpListener`] into
    /// [non-blocking mode][TcpListener::set_nonblocking()], allowing to run
    /// different tasks while waiting for a connection by
    /// [OxiDD-vis](https://oxidd.net/vis) (or another compatible tool).
    ///
    /// Note that you need to call [`poll()`][VisualizationListener::poll()]
    /// repeatedly on the returned [`VisualizationListener`] to accept a TCP
    /// connection.
    pub fn serve_nonblocking(&mut self) -> io::Result<VisualizationListener<'_>> {
        let port = self.port;
        let listener = TcpListener::bind(("localhost", port))?;
        listener.set_nonblocking(true)?;
        println!("Data can be read on http://localhost:{port}");

        Ok(VisualizationListener {
            visualizer: self,
            listener,
            buf: Box::new(EMPTY_RECV_BUF),
        })
    }

    /// Returns `Ok(true)` if the visualization has been sent, `Ok(false)` if
    /// the client did not request `/diagrams`
    fn handle_client(&mut self, stream: TcpStream, buf: &mut RecvBuf) -> io::Result<bool> {
        use io::ErrorKind::*;
        let mut stream = HttpStream::new(stream, buf)?;
        loop {
            return match self.handle_req(&mut stream) {
                Ok(false) => continue,
                Err(e) if matches!(e.kind(), UnexpectedEof | WouldBlock | TimedOut) => Ok(false),
                res => res,
            };
        }
    }

    /// Returns `Ok(true)` if the visualization has been sent, `Ok(false)` if
    /// request is for a path different from `/diagrams`
    fn handle_req(&mut self, stream: &mut HttpStream) -> io::Result<bool> {
        // Basic routing: only handle "GET /diagrams"
        // After the path, there should be "HTTP/1.1" (or something alike),
        // so expecting the space is fine.
        if !stream.next_request_starts_with(b"GET /diagrams ")? {
            stream.conn.write_all(
                b"HTTP/1.1 404 NOT FOUND\r\n\
                Content-Type: text/plain\r\n\
                Content-Length: 9\r\n\
                Access-Control-Allow-Origin: *\r\n\
                \r\n\
                Not Found",
            )?;
            return Ok(false);
        }

        let len = self.buf.len() + 2; // 2 -> account for outer brackets
        write!(
            stream.conn,
            "HTTP/1.1 200 OK\r\n\
            Content-Type: application/json\r\n\
            Content-Length: {len}\r\n\
            Access-Control-Allow-Origin: *\r\n\
            \r\n\
            ["
        )?;
        self.buf.push(b']');
        if let Err(err) = stream.conn.write_all(&self.buf) {
            self.buf.pop();
            return Err(err);
        }
        self.buf.clear();
        println!("Visualization has been sent!");
        Ok(true)
    }
}

type RecvBuf = [u8; 1024];
const EMPTY_RECV_BUF: RecvBuf = [0; 1024];
struct HttpStream<'a> {
    recv_buf: &'a mut RecvBuf,
    read_bytes: usize,
    conn: TcpStream,
}

impl<'a> HttpStream<'a> {
    fn new(stream: TcpStream, recv_buf: &'a mut RecvBuf) -> io::Result<Self> {
        stream.set_nonblocking(false)?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
        Ok(Self {
            recv_buf,
            read_bytes: 0,
            conn: stream,
        })
    }

    /// Read the next HTTP request and check if it starts with the expected byte
    /// sequence (e.g., `GET /foo `).
    fn next_request_starts_with(&mut self, expected: &[u8]) -> io::Result<bool> {
        debug_assert!(expected.len() <= self.recv_buf.len());
        let result = loop {
            // check before read, because we may already have received the
            // request on the last call
            if self.read_bytes >= expected.len() {
                break self.recv_buf[..expected.len()] == *expected;
            }
            self.read_bytes += self.conn.read(&mut self.recv_buf[self.read_bytes..])?;
            if self.read_bytes == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
        };
        // Find the message's end, which is "\r\n\r\n" according to the RFC.
        // Instead, we just test for the newlines.
        let mut last_nl = self.recv_buf.len(); // dummy value
        loop {
            for (i, &c) in self.recv_buf[..self.read_bytes].iter().enumerate() {
                if c == b'\n' {
                    if i == last_nl.wrapping_add(2) {
                        // there may be another message in the buffer, copy it
                        // to the beginning
                        self.recv_buf.copy_within(i + 1..self.read_bytes, 0);
                        self.read_bytes -= i + 1;
                        return Ok(result);
                    }
                    last_nl = i;
                }
            }
            last_nl = last_nl.wrapping_sub(self.read_bytes);
            self.read_bytes = self.conn.read(self.recv_buf)?;
            if self.read_bytes == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
        }
    }
}

/// A non-blocking [`TcpListener`] for decision diagram visualization
pub struct VisualizationListener<'a> {
    visualizer: &'a mut Visualizer,
    listener: TcpListener,
    buf: Box<RecvBuf>,
}

impl VisualizationListener<'_> {
    /// Poll for clients like [OxiDD-vis](https://oxidd.net/vis)
    ///
    /// If a connection was established, this method will directly handle the
    /// client. If the decision diagram(s) were successfully sent, the return
    /// value is `Ok(true)`. If no client was available
    /// ([`TcpListener::accept()`] returned an error with
    /// [`io::ErrorKind::WouldBlock`]), the return value is `Ok(false)`.
    /// `Err(..)` signals a communication error.
    pub fn poll(&mut self) -> io::Result<bool> {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    if self.visualizer.handle_client(stream, &mut self.buf)? {
                        return Ok(true);
                    }
                }
                Err(e) => {
                    return if e.kind() == io::ErrorKind::WouldBlock {
                        Ok(false)
                    } else {
                        Err(e)
                    };
                }
            }
        }
    }
}

#[inline]
fn hex_digit(c: u8) -> u8 {
    if c >= 10 {
        b'a' + c - 10
    } else {
        b'0' + c
    }
}

fn json_escape(target: &mut Vec<u8>, data: &[u8]) {
    for &c in data {
        let mut seq;
        target.extend_from_slice(match c {
            0x08 => b"\\b",
            b'\t' => b"\\t",
            b'\n' => b"\\n",
            0x0c => b"\\f",
            b'\r' => b"\\r",
            b'\\' => b"\\\\",
            b'"' => b"\\\"",
            _ if c < 0x20 => {
                seq = *b"\\u0000";
                seq[4] = hex_digit(c >> 4);
                seq[5] = hex_digit(c & 0b1111);
                &seq
            }
            0x7f => b"\\u007f", // ASCII DEL
            _ => {
                target.push(c);
                continue;
            }
        })
    }
}

struct JsonStrWriter<'a>(&'a mut Vec<u8>);

impl std::io::Write for JsonStrWriter<'_> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        json_escape(self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
