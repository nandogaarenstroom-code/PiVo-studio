#[test]
fn selected_backend_matches_expectation() {
    let selected = pivo_runner::cpu::backend_name();
    eprintln!("selected CPU backend: {selected}");
    if let Ok(expected) = std::env::var("PIVO_EXPECT_CPU_BACKEND") {
        assert_eq!(selected, expected, "requested backend was unavailable or dispatch was wrong");
    }
}
