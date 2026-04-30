<<<<<<< HEAD
# NLPOpt-Net-Experiments
This is the repository for the experiments shown in the Article for NLPOpt-Net framework.
=======
# NLPOpt-Net Experiment Repository

This repository includes the codes and instruction to run the experiments provided in the article to reproduce the results.

The main repository includes the notebook to reproduce the result for Table 01.

## Running the Experiments

To run the experiments you need to call:

python main.py --type 'type' --action 'action' --p 'no of parameter' --n 'no of variables' --me 'no of equality' --mi 'no of inequality' --train_frac 'training fraction'

We excecuted the followings to generate the results provided in the article.

### Results in Table 02
```bash
python main.py --type qp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qcqp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type nlp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type nonconvx --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
```

### Results in Table 03
```bash
python main.py --type qp --action compare_dc3 --p 10 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 25 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 75 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 100 --n 100 --me 50 --mi 50 --train_frac 0.8
```

### Results in Table 04
```bash
python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 10 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 30 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 70 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 90 --train_frac 0.8
```

### Results in Table 05
```bash
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 10 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 30 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 70 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 90 --train_frac 0.8
```

### Results in Table 06

```bash
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.6
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.4
python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.2
```

### Results in Table 07

For the active set agreement, the value is saved in all runs in the `summary.json` file under each problem run directory.

## Citation

```text
@article{bimol2026nlpoptnet,
  title={NLPOpt-Net: A Learning Method for Nonlinear Optimization with Feasibility Guarantees},
  author={Nath Roy, Bimol and Golder, Rahul and Hasan, MM Faruque},
  journal={ },
  year={2026}
  }
```
>>>>>>> 0e1ed0f (first commit)
