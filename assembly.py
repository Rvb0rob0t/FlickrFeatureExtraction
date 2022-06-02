import argparse
import glob
import os


def assembly(input_pathnames, output_pathname):
    with open(output_pathname, 'w') as o:
        o.write('[')
        for p in input_pathnames[:-1]:
            with open(p, 'r') as f:
                o.write(f.read())
                o.write(',\n')
        with open(input_pathnames[-1], 'r') as f:
            o.write(f.read())
        o.write(']\n')


def group_by_depth(pathnames, depth):
    groups = {}
    for p in pathnames:
        group_id = p.split('/')[depth]
        if group_id in groups:
            groups[group_id].append(p)
        else:
            groups[group_id] = [p]
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('glob_pattern',
                        metavar='P',
                        help='pathname pattern of the json files to be assembled')
    parser.add_argument('-o', '--output',
                        help='json output file (directory if -s is used) to contain the json array',
                        required=False,
                        default='assembled.json')
    parser.add_argument('-s', '--separate-by-depth',
                        help=('Path depth at which to separate filenames ',
                              '(can be negative to specify from right to left, -1 is basename)'),
                        required=False)
    args = parser.parse_args()

    json_pathnames = glob.glob(args.glob_pattern)
    if json_pathnames:
        if args.separate_by_depth is None:
            assembly(json_pathnames, args.output)
        else:
            os.makedirs(args.output, exist_ok=True)
            groups = group_by_depth(json_pathnames, int(args.separate_by_depth))
            for group_name, paths in groups.items():
                assembly(paths, os.path.join(args.output, group_name + '.json'))
    else:
        print(f'cannot access {args.glob_pattern}: No such file or directory')


if __name__ == '__main__':
    main()
