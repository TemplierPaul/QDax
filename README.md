# Genetic Drift Regularization

This code is based on a fork of QDax. QDax is a tool to accelerate Quality-Diversity (QD) and neuro-evolution algorithms through hardware accelerators and massive parallelization. QD algorithms usually take days/weeks to run on large CPU clusters. With QDax, QD algorithms can now be run in minutes! ⏩ ⏩ 🕛

More information about QDax can be found in the [QDax readme](qdax_README.md).

## Running Evolution Strategies
ES can be run with Brax environments using the following command:
```
python run_es.py --es=canonical --env=walker2d_uni
```

All parameters can be passed as command line arguments. For a full list of parameters, run `python run_es.py --help` or check out the [qdax.core.rl_es_parts.es_setup](qdax/core/rl_es_parts/es_setup.py) file for the parser definition and default values. 

## Running ES + TD3
### No injection
TD3 can be trained on the transitions from the ES by running:
```
python run_es.py --es=canonical --env=walker2d_uni --rl
```
### Standard injection
Injection can be added with:
```
python run_es.py --es=canonical --env=walker2d_uni --rl --actor_injection
```

### Clipping
Clipping can be added with:
```
python run_es.py --es=canonical --env=walker2d_uni --rl --actor_injection --injection_clip
```

### Genetic drift regularization
GDR can be added by setting a non-zero value to the `--elastic_pull` flag:
```
python run_es.py --es=canonical --env=walker2d_uni --rl --actor_injection --elastic_pull=0.1
```

## Logging 
Logging is handled through [Weights & Biases](https://wandb.ai) by adding a flag:
```
python run_es.py --es=canonical --env=walker2d_uni --wandb=entity/project
```

Two useful values can be logged to the run config by passing them as arguments:
- `--job` for the Slurm job id
- `--tag` for a custom tag to group runs more easily
