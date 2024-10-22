use candle_datasets::vision::Dataset;
use candle_core::{DType, Device, Tensor};
use anyhow::{Ok, Result};
use image;
use parquet::file::reader::SerializedFileReader;
use std::{fs::File, path::Path};


pub fn init_mnist_ds() -> Result<Dataset> {
    // Read in the files
    let train_path = Path::new("data/train-00000-of-00001.parquet");
    let test_path = Path::new("data/test-00000-of-00001.parquet");
    let train_file = File::open(&train_path)?;
    let test_file = File::open(&test_path)?;
    let train_parquet = SerializedFileReader::new(train_file)?;
    let test_parquet = SerializedFileReader::new(test_file)?;
    
    // Straight from https://huggingface.github.io/candle/training/mnist.html
    let test_samples = 10_000;
    let mut test_buffer_images: Vec<u8> = Vec::with_capacity(test_samples * 784);
    let mut test_buffer_labels: Vec<u8> = Vec::with_capacity(test_samples);
    for row in test_parquet{
        for (_name, field) in row?.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        test_buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                test_buffer_labels.push(*label as u8);
            }
        }
    }
    let test_images = (Tensor::from_vec(test_buffer_images, (test_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
    let test_labels = Tensor::from_vec(test_buffer_labels, (test_samples, ), &Device::Cpu)?;

    let train_samples = 60_000;
    let mut train_buffer_images: Vec<u8> = Vec::with_capacity(train_samples * 784);
    let mut train_buffer_labels: Vec<u8> = Vec::with_capacity(train_samples);
    for row in train_parquet{
        for (_name, field) in row?.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        train_buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                train_buffer_labels.push(*label as u8);
            }
        }
    }
    let train_images = (Tensor::from_vec(train_buffer_images, (train_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
    let train_labels = Tensor::from_vec(train_buffer_labels, (train_samples, ), &Device::Cpu)?;

    let mnist = Dataset{
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    };
    
    return Ok(mnist);
}
