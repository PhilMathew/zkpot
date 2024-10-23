## Downloading MNIST files
We use the parquet files provided by huggingface at [this link](https://huggingface.co/datasets/ylecun/mnist/tree/main/mnist). You can either simply run the download script in `download_mnist.sh`, or you can download both the train and test files from the link, then put them into a folder named `data`. In either case, upon completion your directory should be structured as follows:

```
zkpot
|-- data
|   |-- train.parquet
|   |-- test.parquet 
|-- src
|
...
```