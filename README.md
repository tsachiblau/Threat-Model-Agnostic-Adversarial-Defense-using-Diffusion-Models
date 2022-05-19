# Threat-Model-Agnostic-Adversarial-Defense-using-Diffusion-Models

This repo contains the official implementation for the paper [Threat Model-Agnostic Adversarial Defense using
Diffusion Models](http://). 

## Running Experiments

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the ad environment.

```bash
conda env create -f environment.yml
```
The environment includes `cudatoolkit=11.0`. You may change that depending on your hardware.

### Project structure

`main.py` is the file that you should run for eval. Execute ```python main.py --help``` to get its usage description:

```
--arch  trades 
--dataset cifar-c

--first_step
--timesteps

--adv_epsilon
--adv_attack_type
--adv_threat_model

--batch_size 100 


usage: main.py [-h] --arch ARCHITECTURE [--dataset DATASET] [--first_step FIRST_STEP]
               [--timesteps TIME_STEPS] [--adv_epsilon ADVERSARIAL_EPSILON] [--adv_attack_type ADVERSARIAL_ATTACK_TYPE]
               [--adv_threat_model ADVERSARIAL_THREAT_MODEL] [-batch_size BATCH_SIZE] 

optional arguments:
  -h, --help            show this help message and exit
  
  --arch                                                                      | ddim |  adp | at | per | uncovering | trades |
  --dataset cifar-c

  --first_step          First time step of the diffusion model T^{*}          | in range [1, 1000]
  --timesteps           How many time steps will a full depth diffusion have  | in range [1, 1000]

  --adv_epsilon         The adversarial attack norm upper value               
  --adv_attack_type     Which attack do we want to use                        | no_attack | grey | BPDA_EOT | white | white_EOT | 
  --adv_threat_model                                                          | linf | l2 |

  --batch_size          batch size    

```

### Downloading data

it is downloaded automatically

### Running The Code

To evaluate out method under white-box + EOT on CIFAR-10 you should run:
```bash
python main.py --arch ddim --first_step 140 --timesteps 100 --adv_epsilon 0.0313725 --adv_attack_type white_EOT --adv_threat_model linf  --batch_size 10
```
Note that you can choose any other method from the list [adp, at, per, uncovering, trades]



## Pretrained Checkpoints
For our method you should get the diffusion model checkpoint from 
Link: https://github.com/ermongroup/ddim
you should get the classifier checkpoint from 
Link: https://github.com/point0bar1/ebm-defense

For the other methods you should take the checkpoints from their repo
adp           Link: https://github.com/jmyoon1/adp  
at            Link: https://github.com/MadryLab/robustness  
per           Link: https://github.com/cassidylaidlaw/perceptual-advex  
uncovering    Link: https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness  
trades        Link: https://github.com/yaodongyu/TRADES  

## Acknowledgement

This repo is largely based on the [DDIM](https://github.com/ermongroup/ddim) repo
