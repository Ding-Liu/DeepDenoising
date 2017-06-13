"""Train network
"""
import argparse
import caffe
import sys
sys.path.append('./')


def main():
    print sys.path
    parser = argparse.ArgumentParser('Train Network using python')
    parser.add_argument('--solver', required=True,
                        help='Path of solver')
    parser.add_argument('--weights', 
                        help='Denoising model to finetune from, optional')
    parser.add_argument('--weights_2',
                        help='VGG Model to finetune from, optional')
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
    if args.weights_2:
        print('Finetuning from {}'.format(args.weights_2))
        solver.net.copy_from(args.weights_2)
    if args.snapshot:
        print('Resume training from {}'.format(args.snapshot))
        solver.restore(args.snapshot)

    # save
    # solver.net.save('cls_joint_iter0_s30.caffemodel')
    # exit()

    print("Start Training")
    solver.solve()


if __name__ == '__main__':
    main()
