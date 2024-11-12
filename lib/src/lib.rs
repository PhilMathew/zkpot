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


pub struct MLP {
    // W1
    w1_11: u32,
    w1_12: u32,
    w1_13: u32,
    w1_14: u32,
    w1_21: u32,
    w1_22: u32,
    w1_23: u32,
    w1_24: u32,
    w1_31: u32,
    w1_32: u32,
    w1_33: u32,
    w1_34: u32,
    w1_41: u32,
    w1_42: u32,
    w1_43: u32,
    w1_44: u32,

    // b1
    b1_1: u32,
    b1_2: u32,
    b1_3: u32,
    b1_4: u32,

    // W2
    w2_11: u32,
    w2_12: u32,
    w2_13: u32,
    w2_14: u32,
    w2_21: u32,
    w2_22: u32,
    w2_23: u32,
    w2_24: u32,
    w2_31: u32,
    w2_32: u32,
    w2_33: u32,
    w2_34: u32,

    // b2
    b2_1: u32,
    b2_2: u32,
    b2_3: u32,
}

static MLP_TEST: [[Fixed; 7]; 1] = [
    [1.0, 2.0, 3.0, 4.0, 1.0, 0.0, 0.0]
];

pub enum MLP_Dataset {
    MLP_TEST
}

pub fn u32_to_mlp_ds(i: u32) -> MLP_Dataset {
    match i {
        0 => MLP_Dataset::MLP_TEST,
        _ => panic!("Yeah that ain't it chief"),
    }

}

pub fn get_mlp_dataset(ds: MLP_Dataset) -> [[Fixed; 7]; 1] {
    match ds {
        MLP_Dataset::MLP_TEST => MLP_TEST,
        _ => panic!("Yeah that ain't gonna work")
    }
}

pub fn relu(n: Fixed) -> Fixed {
    let zero = ZERO;
    if zero.lt(&n) {
        return n
    } else {
        return ZERO
    }
}

pub fn update_mlp(mlp: MLP, d: u32, num_epochs: u32, eta: u32) -> MLP {
    // Training-specific items
    let dataset = get_mlp_dataset(u32_to_mlp_ds(d));
    let lr = Fixed::from_bits(eta);

    // Model parameters
    // W1
    let mut w1_11 = Fixed::from_bits(mlp.w1_11);
    let mut w1_12 = Fixed::from_bits(mlp.w1_12);
    let mut w1_13 = Fixed::from_bits(mlp.w1_13);
    let mut w1_14 = Fixed::from_bits(mlp.w1_14);
    let mut w1_21 = Fixed::from_bits(mlp.w1_21);
    let mut w1_22 = Fixed::from_bits(mlp.w1_22);
    let mut w1_23 = Fixed::from_bits(mlp.w1_23);
    let mut w1_24 = Fixed::from_bits(mlp.w1_24);
    let mut w1_31 = Fixed::from_bits(mlp.w1_31);
    let mut w1_32 = Fixed::from_bits(mlp.w1_32);
    let mut w1_33 = Fixed::from_bits(mlp.w1_33);
    let mut w1_34 = Fixed::from_bits(mlp.w1_34);
    let mut w1_41 = Fixed::from_bits(mlp.w1_41);
    let mut w1_42 = Fixed::from_bits(mlp.w1_42);
    let mut w1_43 = Fixed::from_bits(mlp.w1_43);
    let mut w1_44 = Fixed::from_bits(mlp.w1_44);
    // b1
    let mut b1_1 = Fixed::from_bits(mlp.b1_1);
    let mut b1_2 = Fixed::from_bits(mlp.b1_2);
    let mut b1_3 = Fixed::from_bits(mlp.b1_3);
    let mut b1_4 = Fixed::from_bits(mlp.b1_4);
    // W2
    let mut w2_11 = Fixed::from_bits(mlp.w2_11);
    let mut w2_12 = Fixed::from_bits(mlp.w2_12);
    let mut w2_13 = Fixed::from_bits(mlp.w2_13);
    let mut w2_14 = Fixed::from_bits(mlp.w2_14);
    let mut w2_21 = Fixed::from_bits(mlp.w2_21);
    let mut w2_22 = Fixed::from_bits(mlp.w2_22);
    let mut w2_23 = Fixed::from_bits(mlp.w2_23);
    let mut w2_24 = Fixed::from_bits(mlp.w2_24);
    let mut w2_31 = Fixed::from_bits(mlp.w2_31);
    let mut w2_32 = Fixed::from_bits(mlp.w2_32);
    let mut w2_33 = Fixed::from_bits(mlp.w2_33);
    let mut w2_34 = Fixed::from_bits(mlp.w2_34);
    // b2
    let mut b2_1 = Fixed::from_bits(mlp.b2_1);
    let mut b2_2 = Fixed::from_bits(mlp.b2_2);
    let mut b2_3 = Fixed::from_bits(mlp.b2_3);



    // Forward pass
    for _ in 0..num_epochs {
        for example in dataset {
            // Get x (input data)
            let x_1 = example[0];
            let x_2 = example[1];
            let x_3 = example[2];
            let x_4 = example[3];

            // Get y (one-hot encoded label)
            let y_1 = example[4];
            let y_2 = example[5];
            let y_3 = example[6];

            // Compute z1
            let z1_1 = (w1_11 * x_1) + (w1_12 * x_2) + (w1_13 * x_3) + (w1_14 * x_4) + b1_1;
            let z1_2 = (w1_21 * x_1) + (w1_22 * x_2) + (w1_23 * x_3) + (w1_24 * x_4) + b1_2;
            let z1_3 = (w1_31 * x_1) + (w1_32 * x_2) + (w1_33 * x_3) + (w1_34 * x_4) + b1_3;
            let z1_4 = (w1_41 * x_1) + (w1_42 * x_2) + (w1_43 * x_3) + (w1_44 * x_4) + b1_4;

            // Compute h
            let h_1 = relu(z1_1);
            let h_2 = relu(z1_2);
            let h_3 = relu(z1_3);
            let h_4 = relu(z1_4);

            // Compute yhat
            let yhat_1 = (w2_11 * h_1) + (w2_12 * h_2) + (w2_13 * h_3) + (w2_14 * h_4) + b2_1;
            let yhat_2 = (w2_21 * h_1) + (w2_22 * h_2) + (w2_23 * h_3) + (w2_24 * h_4) + b2_2;
            let yhat_3 = (w2_31 * h_1) + (w2_32 * h_2) + (w2_33 * h_3) + (w2_34 * h_4) + b2_3;

            // Compute updates
            // W1
            let dLdw1_11 = -2.0 * heaviside(z1_1) * x_1 * ((w2_11 * (y_1 - yhat_1)) + (w2_21 * (y_2 - yhat_2)) + (w2_31 * (y_3 - yhat_3)));
            let dLdw1_12 = -2.0 * heaviside(z1_1) * x_2 * ((w2_11 * (y_1 - yhat_1)) + (w2_21 * (y_2 - yhat_2)) + (w2_31 * (y_3 - yhat_3)));
            let dLdw1_13 = -2.0 * heaviside(z1_1) * x_3 * ((w2_11 * (y_1 - yhat_1)) + (w2_21 * (y_2 - yhat_2)) + (w2_31 * (y_3 - yhat_3)));
            let dLdw1_14 = -2.0 * heaviside(z1_1) * x_4 * ((w2_11 * (y_1 - yhat_1)) + (w2_21 * (y_2 - yhat_2)) + (w2_31 * (y_3 - yhat_3)));
            let dLdw1_21 = -2.0 * heaviside(z1_2) * x_1 * ((w2_12 * (y_1 - yhat_1)) + (w2_22 * (y_2 - yhat_2)) + (w2_32 * (y_3 - yhat_3)));
            let dLdw1_22 = -2.0 * heaviside(z1_2) * x_2 * ((w2_12 * (y_1 - yhat_1)) + (w2_22 * (y_2 - yhat_2)) + (w2_32 * (y_3 - yhat_3)));
            let dLdw1_23 = -2.0 * heaviside(z1_2) * x_3 * ((w2_12 * (y_1 - yhat_1)) + (w2_22 * (y_2 - yhat_2)) + (w2_32 * (y_3 - yhat_3)));
            let dLdw1_24 = -2.0 * heaviside(z1_2) * x_4 * ((w2_12 * (y_1 - yhat_1)) + (w2_22 * (y_2 - yhat_2)) + (w2_32 * (y_3 - yhat_3)));
            let dLdw1_31 = -2.0 * heaviside(z1_3) * x_1 * ((w2_13 * (y_1 - yhat_1)) + (w2_23 * (y_2 - yhat_2)) + (w2_33 * (y_3 - yhat_3)));
            let dLdw1_32 = -2.0 * heaviside(z1_3) * x_2 * ((w2_13 * (y_1 - yhat_1)) + (w2_23 * (y_2 - yhat_2)) + (w2_33 * (y_3 - yhat_3)));
            let dLdw1_33 = -2.0 * heaviside(z1_3) * x_3 * ((w2_13 * (y_1 - yhat_1)) + (w2_23 * (y_2 - yhat_2)) + (w2_33 * (y_3 - yhat_3)));
            let dLdw1_34 = -2.0 * heaviside(z1_3) * x_4 * ((w2_13 * (y_1 - yhat_1)) + (w2_23 * (y_2 - yhat_2)) + (w2_33 * (y_3 - yhat_3)));
            let dLdw1_41 = -2.0 * heaviside(z1_4) * x_1 * ((w2_14 * (y_1 - yhat_1)) + (w2_24 * (y_2 - yhat_2)) + (w2_34 * (y_3 - yhat_3)));
            let dLdw1_42 = -2.0 * heaviside(z1_4) * x_2 * ((w2_14 * (y_1 - yhat_1)) + (w2_24 * (y_2 - yhat_2)) + (w2_34 * (y_3 - yhat_3)));
            let dLdw1_43 = -2.0 * heaviside(z1_4) * x_3 * ((w2_14 * (y_1 - yhat_1)) + (w2_24 * (y_2 - yhat_2)) + (w2_34 * (y_3 - yhat_3)));
            let dLdw1_44 = -2.0 * heaviside(z1_4) * x_4 * ((w2_14 * (y_1 - yhat_1)) + (w2_24 * (y_2 - yhat_2)) + (w2_34 * (y_3 - yhat_3)));
            // b1
            let dLdb1_1 = -2.0 * heaviside(z1_1) * ((w2_11 * (y_1 - yhat_1)) + (w2_21 * (y_2 - yhat_2)) + (w2_31 * (y_3 - yhat_3)));
            let dLdb1_2 = -2.0 * heaviside(z1_2) * ((w2_12 * (y_1 - yhat_1)) + (w2_22 * (y_2 - yhat_2)) + (w2_32 * (y_3 - yhat_3)));
            let dLdb1_3 = -2.0 * heaviside(z1_3) * ((w2_13 * (y_1 - yhat_1)) + (w2_23 * (y_2 - yhat_2)) + (w2_33 * (y_3 - yhat_3)));
            let dLdb1_4 = -2.0 * heaviside(z1_4) * ((w2_14 * (y_1 - yhat_1)) + (w2_24 * (y_2 - yhat_2)) + (w2_34 * (y_3 - yhat_3)));
            // W2
            let dLdw2_11 = -2.0 * (y_1 - yhat_1) * h_1;
            let dLdw2_12 = -2.0 * (y_1 - yhat_1) * h_2;
            let dLdw2_13 = -2.0 * (y_1 - yhat_1) * h_3;
            let dLdw2_14 = -2.0 * (y_1 - yhat_1) * h_4;
            let dLdw2_21 = -2.0 * (y_2 - yhat_2) * h_1;
            let dLdw2_22 = -2.0 * (y_2 - yhat_2) * h_2;
            let dLdw2_23 = -2.0 * (y_2 - yhat_2) * h_3;
            let dLdw2_24 = -2.0 * (y_2 - yhat_2) * h_4;
            let dLdw2_31 = -2.0 * (y_3 - yhat_3) * h_1;
            let dLdw2_32 = -2.0 * (y_3 - yhat_3) * h_2;
            let dLdw2_33 = -2.0 * (y_3 - yhat_3) * h_3;
            let dLdw2_34 = -2.0 * (y_3 - yhat_3) * h_4;
            // b2
            let dLdb2_1 = -2.0 * (y_1 - yhat_1);
            let dLdb2_2 = -2.0 * (y_2 - yhat_2);
            let dLdb2_3 = -2.0 * (y_3 - yhat_3);

            // Update parameters
            // W1
            w1_11 = w1_11 + (lr * dLdw1_11);
            w1_12 = w1_12 + (lr * dLdw1_12);
            w1_13 = w1_13 + (lr * dLdw1_13);
            w1_14 = w1_14 + (lr * dLdw1_14);
            w1_21 = w1_21 + (lr * dLdw1_21);
            w1_22 = w1_22 + (lr * dLdw1_22);
            w1_23 = w1_23 + (lr * dLdw1_23);
            w1_24 = w1_24 + (lr * dLdw1_24);
            w1_31 = w1_31 + (lr * dLdw1_31);
            w1_32 = w1_32 + (lr * dLdw1_32);
            w1_33 = w1_33 + (lr * dLdw1_33);
            w1_34 = w1_34 + (lr * dLdw1_34);
            w1_41 = w1_41 + (lr * dLdw1_41);
            w1_42 = w1_42 + (lr * dLdw1_42);
            w1_43 = w1_43 + (lr * dLdw1_43);
            w1_44 = w1_44 + (lr * dLdw1_44);
            // b1
            b1_1 = b1_1 + (lr * dLdb1_1);
            b1_2 = b1_2 + (lr * dLdb1_2);
            b1_3 = b1_3 + (lr * dLdb1_3);
            b1_4 = b1_4 + (lr * dLdb1_4);
            // W2
            w2_11 = w2_11 + (lr * dLdw2_11);
            w2_12 = w2_12 + (lr * dLdw2_12);
            w2_13 = w2_13 + (lr * dLdw2_13);
            w2_14 = w2_14 + (lr * dLdw2_14);
            w2_21 = w2_21 + (lr * dLdw2_21);
            w2_22 = w2_22 + (lr * dLdw2_22);
            w2_23 = w2_23 + (lr * dLdw2_23);
            w2_24 = w2_24 + (lr * dLdw2_24);
            w2_31 = w2_31 + (lr * dLdw2_31);
            w2_32 = w2_32 + (lr * dLdw2_32);
            w2_33 = w2_33 + (lr * dLdw2_33);
            w2_34 = w2_34 + (lr * dLdw2_34);
            // b2
            b2_1 = b2_1 + (lr * dLdb2_1);
            b2_2 = b2_2 + (lr * dLdb2_2);
            b2_3 = b2_3 + (lr * dLdb2_3);
        }
    }

    MLP{
        w1_11: w1_11.to_bits(),
        w1_12: w1_12.to_bits(),
        w1_13: w1_13.to_bits(),
        w1_14: w1_14.to_bits(),
        w1_21: w1_21.to_bits(),
        w1_22: w1_22.to_bits(),
        w1_23: w1_23.to_bits(),
        w1_24: w1_24.to_bits(),
        w1_31: w1_31.to_bits(),
        w1_32: w1_32.to_bits(),
        w1_33: w1_33.to_bits(),
        w1_34: w1_34.to_bits(),
        w1_41: w1_41.to_bits(),
        w1_42: w1_42.to_bits(),
        w1_43: w1_43.to_bits(),
        w1_44: w1_44.to_bits(),
        b1_1: b1_1.to_bits(),
        b1_2: b1_2.to_bits(),
        b1_3: b1_3.to_bits(),
        b1_4: b1_4.to_bits(),
        w2_11: w2_11.to_bits(),
        w2_12: w2_12.to_bits(),
        w2_13: w2_13.to_bits(),
        w2_14: w2_14.to_bits(),
        w2_21: w2_21.to_bits(),
        w2_22: w2_22.to_bits(),
        w2_23: w2_23.to_bits(),
        w2_24: w2_24.to_bits(),
        w2_31: w2_31.to_bits(),
        w2_32: w2_32.to_bits(),
        w2_33: w2_33.to_bits(),
        w2_34: w2_34.to_bits(),
        b2_1: b2_1.to_bits(),
        b2_2: b2_2.to_bits(),
        b2_3: b2_3.to_bits(),
    }
}