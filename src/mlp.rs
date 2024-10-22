use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarMap, VarBuilder};
use std::path::Path;
use log::info;


pub struct MLP<'a> {
    layer1: Linear,
    layer2: Linear,
    varmap: VarMap,
    weights_path: Option<&'a str>
}


impl <'a> MLP<'a> {
    pub fn new(
        input_size: usize, 
        num_classes: usize, 
        hidden_size: usize, 
        device: &Device, 
        weights_path: Option<&'a str>, 
        load_saved_weights: bool, 
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        match weights_path {
            None => (), // do nothing here; don't want the model to save
            Some(weights_path) => {
                if load_saved_weights && Path::new(&weights_path).exists() { // load from the path if it exists
                    varmap.load(weights_path)?;
                }
            }
        }

        let fc1 = linear(input_size, hidden_size, vb.push_prefix("fc1"))?;
        let fc2 = linear(hidden_size, num_classes, vb.push_prefix("fc2"))?;

        return Ok(
            Self{
                layer1: fc1,
                layer2: fc2,
                varmap: varmap,
                weights_path: weights_path,
            }
        );
    }

    pub fn get_varmap(&self) -> & VarMap {
        return &self.varmap;
    }

    pub fn save_weights(&self) {
        match self.weights_path {
            None => info!("No weights path specified; saving does nothing!"),
            Some(path) => {
                self.varmap.save(path).unwrap();
                info!("Saved weights to {}", path);
            }
        }
    }
}


impl <'a> Module for MLP<'a> {
    fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let img = img.to_dtype(self.layer1.weight().dtype())?;
        let x = self.layer1.forward(&img)?;
        let x = x.relu()?;
        let y = self.layer2.forward(&x)?;
        
        return Ok(y);
    }
}


