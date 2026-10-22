use perf_rusthttp::sse::Framer;
#[test]
fn every_boundary_and_partial_eof() {
    let input =
        "\u{feff}: comment\r\ndata: first\r\ndata: second\r\n\r\ndata: third\r\rdata: ignored\n";
    for split in 0..=input.len() {
        let mut framer = Framer::default();
        let mut events = Vec::new();
        for part in [&input.as_bytes()[..split], &input.as_bytes()[split..]] {
            for &byte in part {
                if let Some(event) = framer.push(byte).unwrap() {
                    events.push(event);
                }
            }
        }
        assert_eq!(events, ["first\nsecond", "third"]);
    }
}
#[test]
fn size_excludes_separators_and_includes_partial_tail() {
    for separator in ["\n", "\r", "\r\n"] {
        for extra in ["", "x"] {
            let input = format!(
                ": {}{separator}data:{extra}{separator}{separator}",
                "x".repeat(1024 * 1024 - 7)
            );
            let mut framer = Framer::default();
            assert_eq!(
                input.bytes().any(|b| framer.push(b).is_err()),
                !extra.is_empty()
            );
        }
    }
    let mut framer = Framer::default();
    assert!(format!("{}x", ":x\n".repeat(512 * 1024))
        .bytes()
        .any(|b| framer.push(b).is_err()));
}

#[test]
fn unicode_fields_and_empty_data() {
    for separator in ["\n", "\r", "\r\n"] {
        let input = format!("\u{feff}: ping{separator}event: custom{separator}id: id{separator}retry: 25{separator}data:  你好{separator}data{separator}{separator}data:{separator}{separator}");
        let mut framer = Framer::default();
        let events: Vec<_> = input
            .bytes()
            .filter_map(|b| framer.push(b).unwrap())
            .collect();
        assert_eq!(events, [" 你好\n", ""]);
    }
    let mut framer = Framer::default();
    let input = [
        b"data: \xff".as_slice(),
        "\u{2028}\u{0085}\u{feff}\n\n".as_bytes(),
    ]
    .concat();
    let events: Vec<_> = input
        .into_iter()
        .filter_map(|b| framer.push(b).unwrap())
        .collect();
    assert_eq!(events, ["\u{fffd}\u{2028}\u{0085}\u{feff}"]);
}

#[test]
fn dispatches_at_blank_cr_and_resets_size_per_event() {
    let mut framer = Framer::default();
    let prefix: Vec<_> = b"data: x\r\r"
        .iter()
        .filter_map(|&b| framer.push(b).unwrap())
        .collect();
    assert_eq!(prefix, ["x"]);
    assert_eq!(framer.push(b'\n').unwrap(), None);
    let large = format!("data: {}\n\n", "x".repeat(1024 * 1024 - 6));
    for _ in 0..2 {
        let events: Vec<_> = large
            .bytes()
            .filter_map(|b| framer.push(b).unwrap())
            .collect();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].len(), 1024 * 1024 - 6);
    }
}
