use candle_core::{DType, IndexOp};
use candle_nn::{ops::{*}, Module, Optimizer};
use candle_datasets::vision::Dataset;
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::{Ok, Result};
use log::warn;

use crate::mlp::MLP;


pub fn train_model<'a>(
    model: &'a MLP, 
    ds: &Dataset, 
    epochs: i32, 
    batch_size: i32, 
    learning_rate: f64, 
    weights_path: Option<&'a str>
) -> Result<&'a MLP> {
    let train_imgs = &ds.train_images;
    let train_labels = &ds.train_labels;
    let num_train_imgs = train_imgs.shape().clone().into_dims()[0] as i32;
    let num_batches = (num_train_imgs as f32 / batch_size as f32).ceil() as i32;

    // Init the optimizer
    let optim_config = candle_nn::ParamsAdamW{
        lr: learning_rate,
        ..Default::default()
    };
    let varmap = model.get_varmap();
    let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), optim_config)?;

    // Run the training loop
    for epoch in 0..epochs {
        let pbar = ProgressBar::new(num_batches as u64);

        pbar.set_style(
            ProgressStyle::with_template(
                "{prefix}: {wide_bar} {pos}/{len} [{elapsed_precise}/{eta_precise}] {msg}"
            ).unwrap()
        );
        pbar.set_prefix(format!("Epoch {}", epoch + 1));

        let mut running_loss = 0.0;
        let mut running_acc  = 0.0;
        let mut completed_batches = 0.0;
        // iterate in batches
        for batch_start in (0..num_train_imgs).step_by(batch_size as usize) {
            // Figure out indices
            let batch_num = batch_start as usize;
            let batch_end = (batch_num + (batch_size as usize)).min(num_train_imgs as usize);
            let num_batch_imgs = (batch_end - batch_num) as f32;

            // Get batch data
            let imgs = train_imgs.i(batch_num..batch_end)?;
            let labels = train_labels.i(batch_num..batch_end)?;

            // forward pass
            let output = model.forward(&imgs)?;
            let pred_labels = softmax(&output, 1)?.argmax(1)?;
            let loss = candle_nn::loss::cross_entropy(&output, &labels)?;

            // backprop
            optimizer.backward_step(&loss)?;

            // logging of loss
            let batch_loss = loss.to_scalar::<f32>()?;
            running_loss += batch_loss;

            // logging of accuracy
            let correct_preds = pred_labels.broadcast_eq(&labels.to_dtype(DType::U32)?)?;
            let num_correct = correct_preds.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
            let batch_acc = num_correct / num_batch_imgs;
            running_acc += batch_acc;

            // Add everything to the progress bar
            let pbar_msg = format!(
                "Loss = {:.4}, Accuracy = {:.4}%", 
                running_loss / (completed_batches + 1.0), 
                running_acc / (completed_batches + 1.0) * 100.0
            );
            pbar.set_message(pbar_msg);
            pbar.inc(1);
            
            // Increment the batch counter
            completed_batches += 1.0;
        }
        println!(); // creates a newline so the progress bar isn't overwritten each epoch   
    }

    // Save the trained model
    match weights_path {
        None => warn!("No weights path specified; model is not being saved!"),
        Some(_) => {
            model.save_weights(weights_path);
        }
    }

    return Ok(model);
}


pub fn test_model<'a>(model: &MLP, ds: &Dataset, batch_size: i32) -> Result<(f32, f32)> {
    let test_imgs = &ds.test_images;
    let test_labels = &ds.test_labels;
    let num_test_imgs = test_imgs.shape().clone().into_dims()[0] as i32;
    let num_batches = (num_test_imgs as f32 / batch_size as f32).ceil() as i32;

    let pbar = ProgressBar::new(num_batches as u64);

    pbar.set_style(
        ProgressStyle::with_template(
            "{prefix}: {wide_bar} {pos}/{len} [{elapsed_precise}/{eta_precise}]"
        ).unwrap()
    );
    pbar.set_prefix("Testing Model");

    let mut running_loss = 0.0;
    let mut num_correct  = 0.0;

    // iterate in batches
    for batch_start in (0..num_test_imgs).step_by(batch_size as usize) {
        // Figure out indices
        let batch_num = batch_start as usize;
        let batch_end = (batch_num + (batch_size as usize)).min(num_test_imgs as usize);

        // Get batch data
        let imgs = test_imgs.i(batch_num..batch_end)?;
        let labels = test_labels.i(batch_num..batch_end)?;

        // forward pass
        let output = model.forward(&imgs)?;
        let pred_labels = softmax(&output, 1)?.argmax(1)?;
        let loss = candle_nn::loss::cross_entropy(&output, &labels)?;

        // logging of loss
        let batch_loss = loss.to_scalar::<f32>()?;
        running_loss += batch_loss;

        // logging of accuracy
        let correct_preds = pred_labels.broadcast_eq(&labels.to_dtype(DType::U32)?)?;
        num_correct += correct_preds.sum_all()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
        pbar.inc(1);
    }

    let loss = running_loss / (num_batches as f32);
    let acc = num_correct / (num_test_imgs as f32);

    return Ok((loss, acc));
}