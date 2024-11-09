//! An end-to-end example of using the SP1 SDK to generate a proof of a program that can be executed
//! or have a core proof generated.
//!
//! You can run this script using the following command:
//! ```shell
//! RUST_LOG=info cargo run --release -- --execute
//! ```
//! or
//! ```shell
//! RUST_LOG=info cargo run --release -- --prove
//! ```

use alloy_sol_types::SolType;
use clap::Parser;
use fibonacci_lib::PublicValuesStruct;
use sp1_sdk::{ProverClient, SP1Stdin};

/// The ELF (executable and linkable format) file for the Succinct RISC-V zkVM.
pub const PERCEPTRON_ELF: &[u8] = include_bytes!("../../../elf/riscv32im-succinct-zkvm-elf");

/// The arguments for the command.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    execute: bool,

    #[clap(long)]
    prove: bool,

    #[clap(long, default_value = "1")]
    d: u32,

    #[clap(long, default_value = "10")]
    t: u32,

    #[clap(long, default_value = "1")]
    r: u32,

    #[clap(long, default_value = "1")]
    w_l: u32,

    #[clap(long, default_value = "1")]
    w_r: u32,
}

fn main() {
    // Setup the logger.
    sp1_sdk::utils::setup_logger();

    // Parse the command line arguments.
    let args = Args::parse();

    if args.execute == args.prove {
        eprintln!("Error: You must specify either --execute or --prove");
        std::process::exit(1);
    }
    
    let r = (0.1_f32).to_bits();
    let w_l = (0.1_f32).to_bits();
    let w_r = (-0.1_f32).to_bits();

    // Setup the prover client.
    let client = ProverClient::new();

    // Setup the inputs.
    let mut stdin = SP1Stdin::new();
    
    stdin.write(&args.d);
    stdin.write(&args.t);
    stdin.write(&r);
    stdin.write(&w_l);
    stdin.write(&w_r);

    println!("d: {}", args.d);
    println!("t: {}", args.t);
    println!("r: {}", r);
    println!("w_l: {}", w_l);
    println!("w_r: {}", w_r);

    if args.execute {
        // Execute the program
        let (output, report) = client.execute(PERCEPTRON_ELF, stdin).run().unwrap();
        println!("Program executed successfully.");

        // Read the output.
        let decoded = PublicValuesStruct::abi_decode(output.as_slice(), true).unwrap();
        let PublicValuesStruct { d, t, r, w_l, w_r, updated_w_l, updated_w_r } = decoded;
        
        println!("d: {:#034b}", d);
        println!("t: {:#034b}", t);
        println!("r: {:#034b}", r);
        println!("w_l: {:#034b}", w_l);
        println!("w_r: {:#034b}", w_r);
        println!("updated_w_l: {:#034b}", updated_w_l);
        println!("updated_w_r: {:#034b}", updated_w_r);

        let (expected_updated_w_l, expected_updated_w_r) = fibonacci_lib::update_perceptron(d, t, r, w_l, w_r);
        assert_eq!(updated_w_l, expected_updated_w_l);
        assert_eq!(updated_w_r, expected_updated_w_r);
        println!("Values are correct!");

        // Record the number of cycles executed.
        println!("Number of cycles: {}", report.total_instruction_count());
    } else {
        // Setup the program for proving.
        let (pk, vk) = client.setup(PERCEPTRON_ELF);

        // Generate the proof
        let proof = client
            .prove(&pk, stdin)
            .run()
            .expect("failed to generate proof");

        println!("Successfully generated proof!");

        // Verify the proof.
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Successfully verified proof!");
    }
}
