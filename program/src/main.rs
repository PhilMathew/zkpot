//! A simple program that takes a number `n` as input, and writes the `n-1`th and `n`th fibonacci
//! number as an output.

// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);

use alloy_sol_types::SolType;
use fibonacci_lib::{update_perceptron, PublicValuesStruct};

pub fn main() {
    // Read an inputs into the training step.
    let d = sp1_zkvm::io::read::<u32>();
    let t = sp1_zkvm::io::read::<u32>();
    let r = sp1_zkvm::io::read::<u32>();
    let w_l = sp1_zkvm::io::read::<u32>();
    let w_r = sp1_zkvm::io::read::<u32>();

    // Compute a weight update.
    let (updated_w_l, updated_w_r) = update_perceptron(d, t, r, w_l, w_r);

    // Encode the public values of the program.
    let bytes = PublicValuesStruct::abi_encode(&PublicValuesStruct { d, t, r, w_l, w_r, updated_w_l, updated_w_r });

    // Commit to the public values of the program. The final proof will have a commitment to all the
    // bytes that were committed to.
    sp1_zkvm::io::commit_slice(&bytes);
}
