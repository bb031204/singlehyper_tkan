import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description='SingleHyper-TKAN Main Entry')
    sub = parser.add_subparsers(dest='command')

    p_train = sub.add_parser('train')
    p_train.add_argument('--config', type=str, default='configs/config.yaml')
    p_train.add_argument('--resume', type=str, default=None)
    p_train.add_argument('--gpu', type=int, default=None)

    p_predict = sub.add_parser('predict')
    p_predict.add_argument('--config', type=str, default='configs/config.yaml')
    p_predict.add_argument('--checkpoint', type=str, default=None)
    p_predict.add_argument('--output_dir', type=str, default=None)
    p_predict.add_argument('--gpu', type=int, default=None)

    p_all = sub.add_parser('all')
    p_all.add_argument('--config', type=str, default='configs/config.yaml')
    p_all.add_argument('--gpu', type=int, default=None)

    args = parser.parse_args()

    if args.command == 'train':
        cmd = ['python', 'train.py', '--config', args.config]
        if args.resume:
            cmd += ['--resume', args.resume]
        if args.gpu is not None:
            cmd += ['--gpu', str(args.gpu)]
        subprocess.run(cmd)
    elif args.command == 'predict':
        cmd = ['python', 'predict.py', '--config', args.config]
        if args.checkpoint:
            cmd += ['--checkpoint', args.checkpoint]
        if args.output_dir:
            cmd += ['--output_dir', args.output_dir]
        if args.gpu is not None:
            cmd += ['--gpu', str(args.gpu)]
        subprocess.run(cmd)
    elif args.command == 'all':
        cmd1 = ['python', 'train.py', '--config', args.config]
        if args.gpu is not None:
            cmd1 += ['--gpu', str(args.gpu)]
        subprocess.run(cmd1)
        cmd2 = ['python', 'predict.py', '--config', args.config]
        if args.gpu is not None:
            cmd2 += ['--gpu', str(args.gpu)]
        subprocess.run(cmd2)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
