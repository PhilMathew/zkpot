mod mlp;
mod mnist;
mod model_utils;

use candle_core::Device;
use mlp::MLP;
use mnist::init_mnist_ds;
use model_utils::{train_model, test_model};


fn main() {
    let device = Device::Cpu;
    let mnist_ds = init_mnist_ds().unwrap();
    let model = MLP::new(784, 10, 64, &device, Some("mlp_mnist.safetensors"), false).unwrap();
    train_model(&model, &mnist_ds, 2, 64, 1e-3).unwrap();
    let (loss, acc) = test_model(&model, &mnist_ds, 64).unwrap();
    println!(
        "Test Loss: {:.4}, Test Accuracy: {:.4}%",
        loss,
        acc * 100.0
    );
}
