## Introduction
Code for "Beyond Marginal Uncertainty: How Accurately can Bayesian Regression Models Estimate Posterior Predictive Correlations?"
In this paper, we consider the problem of bench- marking how accurately Bayesian models can estimate predictive correlations. 

1. When the ORACLE model is available: Meta-Correlations
2. Tranductive Active Learning
3. Cross-Normalized Joint Likelihoods (XLL, XLLR)

Firstly, through toy problems whose ORACLE PPCs are known, we validate the effectiveness of both TAL and XLL(R). Then we evaluate 
**_How Accurately can Bayesian Regression Models Estimate PPCs_** in real-world problems.

### TOY Problems
We generated synthetic datasets using a Gaussian process whose kernel was obtained from the limiting distribution of 
infinitely wide Bayesian ReLU network with one hidden layer. To generate the data, use
```
    python tasks/toy/generate_toy_data.py
```

The generated data comes with the ORACLE posterior predictive variances and correlations, from which we can compute the 
meta-correlations to evaluate the predictive uncertainty of our models. We use meta-correlations to validate TAL and XLL(R). 
For example, the following command computes and saves the meta-corrs, TAL scores and XLLs for the GP model. 
```
    python main.py --task toy_compute_metrics --method gp --input_dim 9
```
Having the results for a list of models, we can visualize whether meta-corrs, TAL scores and XLLs are strongly correlated
 using `notebooks/scatter_metrics.py`.


### UCI Problems

#### Hyperparameter Selection
For each model and each dataset, we run standard regression problems and use the validation lld to select the hyperparameter.

#### Tranductive Active Learning (TAL)
Fortunately, the selection process and the prediction process in TAL are decoupled. Thus we can run multiple successive selections
without incuring the prediction model. Having done the selection for all iterations, we then can evaluate them using any prediction model.
A sample code for the selection process is,  
```
    python main.py --task al_selection --method gp --criteria batchMIG --dataset boston
```
We can also use the ORACLE model for selection. Before using the ORACLE model, we firstly pretrain its kernel using the train+pool set.
```
    python main.py --task al_oracle_pretrain --method gp --dataset boston \
    --epochs 5000 --n_base 1000 --batch_size 2000 --kernel nkn -ard
```
Then we can use the pre-trained ORACLE for the selection process as well,
```
    python main.py --task al_oracle_selection --method gp --criteria batchMIG --dataset boston \
    --epochs 2000 --n_base 1000 --batch_size 2000 --kernel nkn -ard 
```

After the selection process, the prediction process generates the reported performances,
```
    python main.py --task al_prediction --method gp --dataset boston \
    --criteria batchMIG --base_method bnn_factorial

    python main.py --task al_oracle_prediction --method gp --dataset boston \
    --criteria batchMIG --base_method bnn_factorial \
    --epochs 2000 --n_base 1000 --batch_size 2000 --kernel nkn -ard 
    
```
Having the predictions, we can visualize the TAL performances using `notebooks/TAL_oracle_prediction.py` and `notebooks/TAL_oracle_criterions.py`.

#### Cross-Normalized Joint Likelihoods
Evaluating the XLL and XLLR requires only the predictive covariance for the test set, which can be saved after training.
```
    python main.py --task regression -rv 0.6 --method gp --dataset boston
```
Having the predictive covariances for different models, we can compute XLL and XLLR using `notebooks/uci_xll.py`. The test predictions of our models can be downloaded from [xll](https://drive.google.com/file/d/1zJBxkwb3QcrG-k_PGDk6P2YpV488jcWz/view?usp=sharing), together with which you can compute XLL(R) for your own model.

#### Scripts
Finally, you can refer to `scripts` for all experiments. To be noted, the examplar codes above are only minimal. For the exact command compatible to our plotting notebooks, please refer to the `scripts`. 
