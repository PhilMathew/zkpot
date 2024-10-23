use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarMap, VarBuilder};
use std::path::Path;
use log::{info, warn};


pub struct MLP {
    layer1: Linear,
    layer2: Linear,
    varmap: VarMap,
}


impl <'a> MLP {
    pub fn new(
        input_size: usize, 
        num_classes: usize, 
        hidden_size: usize, 
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let fc1 = linear(input_size, hidden_size, vb.push_prefix("fc1"))?;
        let fc2 = linear(hidden_size, num_classes, vb.push_prefix("fc2"))?;

        return Ok(
            Self{
                layer1: fc1,
                layer2: fc2,
                varmap: varmap,
            }
        );
    }

    pub fn get_varmap(&self) -> & VarMap {
        return &self.varmap;
    }

    pub fn load_weights(&mut self, weights_path: Option<&'a str>) {
        match weights_path {
            None => warn!("No weights path specified; loading does nothing!"),
            Some(path) => {
                if Path::new(&path).exists() { // load from the path if it exists
                    self.varmap.load(path).unwrap();
                    info!("Loaded weights from {}", path);
                } else {
                    warn!("No file exists at {}; loading will do nothing!", path);
                }
            }
        }
    }

    pub fn save_weights(&self, weights_path: Option<&'a str>) {
        match weights_path {
            None => warn!("No weights path specified; saving does nothing!"),
            Some(path) => {
                self.varmap.save(path).unwrap();
                info!("Saved weights to {}", path);
            }
        }
    }
}


impl <'a> Module for MLP {
    fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let img = img.to_dtype(self.layer1.weight().dtype())?;
        let x = self.layer1.forward(&img)?;
        let x = x.relu()?;
        let y = self.layer2.forward(&x)?;
        
        return Ok(y);
    }
}


