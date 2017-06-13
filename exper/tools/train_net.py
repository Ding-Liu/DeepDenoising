"""Train network
"""
import argparse
import sys
sys.path.append('../python')
import caffe
sys.path.append('./')


def main():
    print sys.path
    parser = argparse.ArgumentParser('Train Network using python')
    parser.add_argument('--solver', required=True,
                        help='Path of solver')
    parser.add_argument('--weights', 
                        help='Model to finetune from, optional')
    parser.add_argument('--snapshot', 
                        help='Snapshot for resume training')
    parser.add_argument('--GPU', help='The GPU id used to train')

    args = parser.parse_args()

    if args.GPU is not None:
        gpu_id = int(args.GPU)
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        caffe.set_mode_cpu()

    solver = caffe.SGDSolver(args.solver)
    if args.weights:
        print('Finetuning from {}'.format(args.weights))
        solver.net.copy_from(args.weights)
    if args.snapshot:
        print('Resume training from {}'.format(args.snapshot))
        solver.restore(args.snapshot)
    print("Start Training")
    solver.solve()


if __name__ == '__main__':
    main()
