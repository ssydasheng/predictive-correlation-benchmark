import os.path as osp
import argparse

parser = argparse.ArgumentParser('script')
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-pn', '--parallel_runs', type=int, default=4)
args = parser.parse_args()


def write_cpu_cmds(f, cmds):
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --partition=cpu\n')
    f.write('#SBATCH --mem=20G\n')
    f.write('#SBATCH --gres=gpu:0\n')
    f.write("#SBATCH --qos=nopreemption\n")
    f.write('#SBATCH --array=0-{}%{}\n'.format(len(cmds) - 1, args.parallel_runs))
    log_file = 'logs/tune-%A_%a.log'
    f.write('#SBATCH --output={}\n'.format(log_file))
    f.write('list=(\n')
    for i, cmd in enumerate(cmds):
        f.write('  "{}"\n'.format(cmd))
    f.write(')\n')
    f.write('${list[SLURM_ARRAY_TASK_ID]}')


def write_gpu_cmds(f, cmds):
    f.write('#!/bin/bash\n')
    f.write('#SBATCH --partition=p100\n')
    f.write('#SBATCH --mem=20G\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write("#SBATCH --qos=nopreemption\n")
    f.write('#SBATCH --array=0-{}%{}\n'.format(len(cmds) - 1, args.parallel_runs))
    log_file = 'logs/tune-%A_%a.log'
    f.write('#SBATCH --output={}\n'.format(log_file))
    f.write('list=(\n')
    for i, cmd in enumerate(cmds):
        f.write('  "{}"\n'.format(cmd))
    f.write(')\n')
    f.write('${list[SLURM_ARRAY_TASK_ID]}')

def toy_metrics(name):

    cmds = []

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method homo_ensemble '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [3,  10,  100]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix +  '--batch_size 100 -n_networks {} -nu {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method bbb '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [3,  10,  100]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix +  '--batch_size 100 --eval_cov_samples {} -nu {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method nng '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [3,  10,  100]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix + '-Lae 500 -Lar 0.1 --batch_size 100 --eval_cov_samples {} -nu {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method dropout '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [3,  10,  100]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix + '--dropout_rate 0.05 --dropout_tau 0.025 --batch_size 100 --eval_cov_samples {} -nu {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method emhmc '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [3,  10,  100]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix + '--hmc_num_chains 1  --hmc_leap_frog_steps 5 --hmc_step_size 0.01 --num_steps_between_samples 100 --opt_eta --init_eta 1.0 --hmc_n_steps_per_em 10 --hmc_n_steps_adapt_step_size 3 --batch_size -1 --hmc_samples {} -nu {} -lr {} --hmc_burnin {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)

    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method gp '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [1]:
            for EP in [100, 300, 1000]:
                for NU in [1]:
                    cmd = prefix + '--eval_cov_samples {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)


    prefix = 'python main.py --task toy_compute_metrics --input_dim 9 --method fbnn '
    for LR in [0.001, 0.003, 0.01]:
        for VAR in [1]:
            for EP in [100, 300, 1000]:
                for NU in [50, 100]:
                    cmd = prefix + '--batch_size -1 --n_rand 2 --eval_cov_samples {} -nu {} -lr {} -e {} --note VAR{}_NU{}_LR{}_EP{}'.format(
                        VAR, NU, LR, EP, VAR, NU, LR, EP,
                    )
                    cmds.append(cmd)


    with open('%s.sh'%name, 'w') as f:
        write_cpu_cmds(f, cmds)


UCI_HYPERS = dict(
    ensemble=dict(
        boston=dict(lr=0.003, nu=400),
        concrete=dict(lr=0.01, nu=50),
        energy=dict(lr=0.01, nu=50),
        wine=dict(lr=0.001, nu=50),
        yacht=dict(lr=0.01, nu=50),
        kin8nm=dict(lr=0.01, nu=400),
        naval=dict(lr=0.001, nu=50),
        power_plant=dict(lr=0.003, nu=400),
    ),
    bbb=dict(
        boston=dict(lr=0.003, nu=400),
        concrete=dict(lr=0.01, nu=50),
        energy=dict(lr=0.01, nu=50),
        wine=dict(lr=0.001, nu=50),
        yacht=dict(lr=0.01, nu=50),
        kin8nm=dict(lr=0.001, nu=400),
        naval=dict(lr=0.003, nu=50),
        power_plant=dict(lr=0.003, nu=50),
    ),
    nng=dict(
        boston=dict(lr=0.003, nu=50),
        concrete=dict(lr=0.001, nu=50),
        energy=dict(lr=0.01, nu=50),
        wine=dict(lr=0.001, nu=50),
        yacht=dict(lr=0.01, nu=50),
        kin8nm=dict(lr=0.001, nu=50),
        naval=dict(lr=0.001, nu=50),
        power_plant=dict(lr=0.003, nu=400),
    ),
    dropout=dict(
        boston=dict(lr=0.01, nu=400, dr=0.05, dt=0.025),
        concrete=dict(lr=0.001, nu=400, dr=0.05, dt=0.025),
        energy=dict(lr=0.003, nu=400, dr=0.0025, dt=0.005),
        wine=dict(lr=0.001, nu=400, dr=0.05, dt=0.125),
        yacht=dict(lr=0.01, nu=400, dr=0.0025, dt=0.005),
        kin8nm=dict(lr=0.001, nu=400, dr=0.01, dt=0.025),
        naval=dict(lr=0.001, nu=400, dr=0.0025, dt=0.005),
        power_plant=dict(lr=0.001, nu=400, dr=0.05, dt=0.025),
    ),
    emhmc=dict(
        boston=dict(lr=0.01, nu=50),
        concrete=dict(lr=0.01, nu=50),
        energy=dict(lr=0.003, nu=50),
        wine=dict(lr=0.001, nu=50),
        yacht=dict(lr=0.001, nu=50),
        kin8nm=dict(lr=0.003, nu=400),
        naval=dict(lr=0.001, nu=400),
        power_plant=dict(lr=0.003, nu=50),
    ),
    gp=dict(
        boston=dict(),
        concrete=dict(),
        energy=dict(),
        wine=dict(),
        yacht=dict(),
        kin8nm=dict(),
        naval=dict(),
        power_plant=dict(),
    ),
    fbnn=dict(
        boston=dict(bs=5000, lr=0.003, nu=400, nr=100),
        concrete=dict(bs=5000, lr=0.01, nu=400, nr=5),
        energy=dict(bs=5000, lr=0.001, nu=400, nr=5),
        wine=dict(bs=5000, lr=0.01, nu=400, nr=5),
        yacht=dict(bs=5000, lr=0.001, nu=400, nr=5),
        kin8nm=dict(bs=900, lr=0.003, nu=400, nr=20),
        naval=dict(bs=900, lr=0.001, nu=400, nr=20),
        power_plant=dict(bs=900, lr=0.01, nu=400, nr=5),
    ),
)


def xll(name):

    cmds = []

    for dataset in ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']:

        # prefix = 'python main.py --task regression --dataset %s --method ensemble ' % dataset
        # cmd = prefix +  '--batch_size 100 -n_networks 100 -e 10000 -nu {} -lr {}'.format(
        #     UCI_HYPERS['ensemble'][dataset]['nu'], UCI_HYPERS['ensemble'][dataset]['lr']
        # )
        # cmds.append(cmd)
        #
        # prefix = 'python main.py --task regression --dataset %s --method bbb ' % dataset
        # cmd = prefix +  '--batch_size 100 --eval_cov_samples 5000 -e 10000 -nu {} -lr {}'.format(
        #     UCI_HYPERS['bbb'][dataset]['nu'], UCI_HYPERS['bbb'][dataset]['lr']
        # )
        # cmds.append(cmd)
        #
        # prefix = 'python main.py --task regression --dataset %s --method nng ' % dataset
        # cmd = prefix + '-Lae 5000 -Lar 0.1 --batch_size 100 --eval_cov_samples 5000 -e 10000 -nu {} -lr {}'.format(
        #     UCI_HYPERS['nng'][dataset]['nu'], UCI_HYPERS['nng'][dataset]['lr']
        # )
        # cmds.append(cmd)

        prefix = 'python main.py --task regression --dataset %s --method dropout ' % dataset
        cmd = prefix + ' --batch_size 100 --eval_cov_samples 5000 -e 10000 --dropout_rate {} --dropout_tau {} -nu {} -lr {} '.format(
            UCI_HYPERS['dropout'][dataset]['dr'], UCI_HYPERS['dropout'][dataset]['dt'],
            UCI_HYPERS['dropout'][dataset]['nu'], UCI_HYPERS['dropout'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task regression --dataset %s --method emhmc ' % dataset
        cmd = prefix + '--hmc_num_chains 10  --hmc_leap_frog_steps 5 --hmc_step_size 0.01 --num_steps_between_samples 100 --opt_eta --init_eta 1.0 --hmc_n_steps_per_em 5 --hmc_n_steps_adapt_step_size 5 --batch_size -1 --hmc_samples 100 --hmc_burnin 5999 -nu {} -lr {} '.format(
            UCI_HYPERS['emhmc'][dataset]['nu'], UCI_HYPERS['emhmc'][dataset]['lr']
        )
        cmds.append(cmd)
        #
        # prefix = 'python main.py --task regression --dataset %s --method gp ' % dataset
        # cmd = prefix + '-e 10000 --n_base 1000 --batch_size 5000 -lr 0.003 -ard'
        # cmds.append(cmd)
        #
        # prefix = 'python main.py --task regression --dataset %s --method fbnn ' % dataset
        # cmd = prefix + ' --batch_size {} -e 10000 -nu {} -lr {} --n_rand {} -ard'.format(
        #     UCI_HYPERS['fbnn'][dataset]['bs'], UCI_HYPERS['fbnn'][dataset]['nu'],
        #     UCI_HYPERS['fbnn'][dataset]['lr'], UCI_HYPERS['fbnn'][dataset]['nr']
        # )
        # cmds.append(cmd)


    with open('%s.sh'%name, 'w') as f:
        write_gpu_cmds(f, cmds)


def al_selection(name):

    cmds = []

    for dataset in ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']:

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method ensemble ' % dataset
        cmd = prefix +  '--batch_size 100 -n_networks 100 -e 10000 -nu {} -lr {}'.format(
            UCI_HYPERS['ensemble'][dataset]['nu'], UCI_HYPERS['ensemble'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method bbb ' % dataset
        cmd = prefix +  '--batch_size 100 --eval_cov_samples 5000 -e 10000 -nu {} -lr {}'.format(
            UCI_HYPERS['bbb'][dataset]['nu'], UCI_HYPERS['bbb'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method nng ' % dataset
        cmd = prefix + '-Lae 5000 -Lar 0.1 --batch_size 100 --eval_cov_samples 5000 -e 10000 -nu {} -lr {}'.format(
            UCI_HYPERS['nng'][dataset]['nu'], UCI_HYPERS['nng'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method dropout ' % dataset
        cmd = prefix + ' --batch_size 100 --eval_cov_samples 5000 -e 10000 --dropout_rate {} --dropout_tau {} -nu {} -lr {} '.format(
            UCI_HYPERS['dropout'][dataset]['dr'], UCI_HYPERS['dropout'][dataset]['dt'],
            UCI_HYPERS['dropout'][dataset]['nu'], UCI_HYPERS['dropout'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method emhmc ' % dataset
        cmd = prefix + '--hmc_num_chains 10  --hmc_leap_frog_steps 5 --hmc_step_size 0.01 --num_steps_between_samples 100 --opt_eta --init_eta 1.0 --hmc_n_steps_per_em 5 --hmc_n_steps_adapt_step_size 5 --batch_size -1 --hmc_samples 100 --hmc_burnin 5999 -nu {} -lr {} '.format(
            UCI_HYPERS['emhmc'][dataset]['nu'], UCI_HYPERS['emhmc'][dataset]['lr']
        )
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method gp ' % dataset
        cmd = prefix + '-e 10000 --n_base 1000 --batch_size 5000 -lr 0.003 -ard'
        cmds.append(cmd)

        prefix = 'python main.py --task al_selection --criteria batchMIG  --dataset %s --method fbnn ' % dataset
        cmd = prefix + ' --batch_size {} -e 10000 -nu {} -lr {} --n_rand {} -ard'.format(
            UCI_HYPERS['fbnn'][dataset]['bs'], UCI_HYPERS['fbnn'][dataset]['nu'],
            UCI_HYPERS['fbnn'][dataset]['lr'], UCI_HYPERS['fbnn'][dataset]['nr']
        )
        cmds.append(cmd)

    with open('%s.sh'%name, 'w') as f:
        write_gpu_cmds(f, cmds)


def al_oracle_pretrain(name):
    cmds = []
    for dataset in ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']:
        cmd = 'python main.py --task al_oracle_pretrain --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 5000' % dataset
        cmds.append(cmd)

    with open('%s.sh'%name, 'w') as f:
        write_gpu_cmds(f, cmds)


def al_oracle_selection(name):
    cmds = []
    for dataset in ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']:
        cmd = 'python main.py --task al_oracle_selection -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000  -bs 2000 -e 2000' % dataset
        cmds.append(cmd)

    with open('%s.sh'%name, 'w') as f:
        write_gpu_cmds(f, cmds)


def al_oracle_prediction(name):
    cmds = []
    for dataset in ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']:
        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method ensemble' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method bbb' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method nng' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method dropout' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method emhmc' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method gp' % dataset
        cmds.append(cmd)

        cmd = 'python main.py --task al_oracle_prediction -criteria batchMIG --method gp --dataset %s --kernel nkn -ard --n_base 1000 -bs 2000 -e 2000 --base_method fbnn' % dataset
        cmds.append(cmd)

    with open('%s.sh'%name, 'w') as f:
        write_gpu_cmds(f, cmds)

if __name__ == '__main__':
    # toy_metrics('scripts/toy_compute_metrics')
    xll('scripts/xll')
    # al_selection('scripts/al_section')
    # al_oracle_pretrain('scripts/al_oracle_pretrain')
    # al_oracle_selection('scripts/al_oracle_selection')
    # al_oracle_prediction('scripts/al_oracle_prediction')
