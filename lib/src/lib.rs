use alloy_sol_types::sol;
//use fixed::types::I16F16;

type Fixed = f32;

const ZERO: f32 = 0.0;
const ONE: f32 = 1.0;

pub enum Dataset {
    AND,
    NAND,
    OR,
    NOR,
    XOR
}

static AND: [[Fixed; 3]; 4] = [
    [ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO],
    [ONE, ZERO, ZERO],
    [ONE, ONE, ONE],
];

static NAND: [[Fixed; 3]; 4] = [
    [ZERO, ZERO, ONE],
    [ZERO, ONE, ONE],
    [ONE, ZERO, ONE],
    [ONE, ONE, ZERO],
];

static OR: [[Fixed; 3]; 4] = [
    [ZERO, ZERO, ZERO],
    [ZERO, ONE, ONE],
    [ONE, ZERO, ONE],
    [ONE, ONE, ONE],
];

static NOR: [[Fixed; 3]; 4] = [
    [ZERO, ZERO, ONE],
    [ZERO, ONE, ZERO],
    [ONE, ZERO, ZERO],
    [ONE, ONE, ZERO],
];

static XOR: [[Fixed; 3]; 4] = [
    [ZERO, ZERO, ZERO],
    [ZERO, ONE, ONE],
    [ONE, ZERO, ONE],
    [ONE, ONE, ZERO],
];

pub fn u32_to_dataset(i: u32) -> Dataset {
    match i {
        0 => Dataset::AND,
        1 => Dataset::NAND,
        2 => Dataset::OR,
        3 => Dataset::NOR,
        4 => Dataset::XOR,

        // If we are not given a valid dataset, return AND by default
        _ => Dataset::AND
    }
}

pub fn get_dataset(dataset: Dataset) -> [[Fixed; 3]; 4] {
    match dataset {
        Dataset::AND => AND,
        Dataset::NAND => NAND,
        Dataset::OR => OR,
        Dataset::NOR => NOR,
        Dataset::XOR => XOR
    }
}

sol! {
    /// The public values encoded as a struct that can be easily deserialized inside Solidity.
    struct PublicValuesStruct {
        uint32 d;
        uint32 t;
        uint32 w_l;
        uint32 w_r;
        uint32 r;

        uint32 updated_w_l;
        uint32 updated_w_r;
    }
}

pub fn heaviside(n: Fixed) -> Fixed {
    let zero = ZERO;

    if zero.lt(&n) {
        return ZERO
    } else {
        return ONE
    }
}

// Train t epochs on the provided dataset, the learning rate, and the given weights.
pub fn update_perceptron(d: u32, t: u32, r: u32, w_l: u32, w_r: u32) -> (u32, u32) {
    let dataset = get_dataset(u32_to_dataset(d));
    let learning_rate = Fixed::from_bits(r);
    let mut weight_left = Fixed::from_bits(w_l);
    let mut weight_right = Fixed::from_bits(w_r);

    for _ in 0..t {
        for example in dataset {
            
            // Unpack the dataset and model
            let left = example[0];
            let right = example[1];
            let bias = ONE;

            let truth = example[2];

            // Run a model prediction
            let predicted = heaviside(weight_left * left + weight_right * right + bias);

            // Update weights
            weight_left = weight_left - learning_rate * (predicted - truth) * left;
            weight_right = weight_right - learning_rate * (predicted - truth) * right;
        }
    }
    
    return (weight_left.to_bits(), weight_right.to_bits())
}
