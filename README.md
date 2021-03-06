# YALAL - Yet Another Linear Algebra Library
A simple linear algebra library written in Rust, featuring:
- Vectors in 2,3, and N dimensions
- Matrices

## Usage
Before I publish this to [crates.io](https://crates.io/), I want to make sure it's presentable. So for now, the crate must be cloned and then put in the root directory of your project. Then, you can add it to your `Cargo.toml` like this:
```toml
[dependencies]
yalal = { version = '0.1.2', path = "./yalal" }
```

## Documentation
If you need documentation, it can be generated by running
```bash
cargo doc --open
```
in the YALAL root directory.
There isn't a lot of documentation currently on each of the functions.
The documentation will be fixed/available once this crate is published to [crates.io](https://crates.io/).
