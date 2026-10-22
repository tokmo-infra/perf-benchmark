use crate::DecodeError;

// Dispatch only at blank lines, discard partial events at EOF. The limit counts
// field/comment bytes, excluding separators, matching the measurement contract.
#[derive(Default)]
pub struct Framer {
    line: Vec<u8>,
    fields: Vec<String>,
    size: usize,
    after_cr: bool,
    seen_line: bool,
}
impl Framer {
    pub fn push(&mut self, byte: u8) -> Result<Option<String>, DecodeError> {
        if self.after_cr {
            self.after_cr = false;
            if byte == b'\n' {
                return Ok(None);
            }
        }
        if byte != b'\n' && byte != b'\r' {
            self.line.push(byte);
            if self.size + self.line.len() > 1024 * 1024 {
                return Err(DecodeError::Protocol);
            }
            return Ok(None);
        }
        self.after_cr = byte == b'\r';
        self.size += self.line.len();
        let line = String::from_utf8_lossy(&self.line);
        let line = if self.seen_line {
            &line
        } else {
            line.strip_prefix('\u{feff}').unwrap_or(&line)
        };
        self.seen_line = true;
        let mut event = None;
        if line.is_empty() {
            self.size = 0;
            if !self.fields.is_empty() {
                event = Some(self.fields.join("\n"));
                self.fields.clear();
            }
        } else {
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            if field == "data" {
                self.fields
                    .push(value.strip_prefix(' ').unwrap_or(value).to_owned());
            }
        }
        self.line.clear();
        Ok(event)
    }
}
