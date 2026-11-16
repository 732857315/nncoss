fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args
        .iter()
        .any(|argument| argument == "-h" || argument == "--help")
    {
        println!("{}", nncoss_web::usage());
        return;
    }
    if let Err(error) = nncoss_web::run_from_args(args) {
        eprintln!("nncoss-web: {error}");
        std::process::exit(2);
    }
}
