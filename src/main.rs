mod mlp;
mod mnist;
mod model_utils;

use candle_core::Device;
use mlp::MLP;
use mnist::init_mnist_ds;
use model_utils::{train_model, test_model};


const LOAD_EXISTING_WEIGHTS: bool = false;
const MODEL_WEIGHTS_PATH: Option<&str> = Some("mlp_mnist.safetensors");


fn main() {
    let device = Device::Cpu;
    let mnist_ds = init_mnist_ds().unwrap();

    let mut model = MLP::new(784, 10, 64, &device).unwrap();
    if LOAD_EXISTING_WEIGHTS {
        model.load_weights(MODEL_WEIGHTS_PATH);
    }
    train_model(&model, &mnist_ds, 2, 64, 1e-3, MODEL_WEIGHTS_PATH).unwrap();
    
    let (loss, acc) = test_model(&model, &mnist_ds, 64).unwrap();
    println!(
        "Test Loss: {:.4}, Test Accuracy: {:.4}%",
        loss,
        acc * 100.0
    );
}
