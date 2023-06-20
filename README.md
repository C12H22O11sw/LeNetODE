# Usage

## Optional Parameters
|parameter         | default value  |  meaning  |
|------------------|----------------|-----------|
|--batch-size      |64              | training set batch size |
|--test-batch-size |1000            | testing set batch size |
|--epochs          |64              | number of epochs | 
|--lr              |1.0             | learning rate | 
|--epsilon         | 0.1 | maximum norm of adversarial attack perturbations | 
|--alpha           | 0.5 | prioritization of standard image loss vs adversarial image loss | 
|--uses-ODE        | True | ODE+LeNet or LeNet | 
|--ode-channels    | 12 | number of channels in the ODE block (must be a multiple of 4!!!) | 
|--dataset         | mnist              | dataset to use (either "mnist" or "fashionMNIST")| 

## Example usage
```bash
python main.py --lr 1e-5 --epsilon 0.3 --epochs 20 --ode-channels 24 --uses-ODE False
```
