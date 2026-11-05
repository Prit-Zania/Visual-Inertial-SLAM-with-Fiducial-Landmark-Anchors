import argparse
import json
from pathlib import Path
from .pipeline import process
from .dataset import import_phone
from .evaluate import evaluate


def main():
    p=argparse.ArgumentParser(description='OnePlus MSCKF + IMU + natural features + ArUco')
    sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('process');run.add_argument('dataset');run.add_argument('--config',required=True)
    run.add_argument('--map');run.add_argument('--baseline',action='store_true');run.add_argument('--no-smooth',action='store_true')
    run.add_argument('--allow-uncalibrated',action='store_true');run.add_argument('--output')
    imp=sub.add_parser('import');imp.add_argument('phone_jsonl');imp.add_argument('--output',default='recordings')
    ev=sub.add_parser('evaluate');ev.add_argument('trajectory');ev.add_argument('ground_truth');ev.add_argument('--align',action='store_true')
    args=p.parse_args()
    if args.command=='process':
        output,summary=process(args.dataset,args.config,args.map,not args.baseline,not args.no_smooth,args.allow_uncalibrated,args.output)
        print(json.dumps(summary,indent=2));print('Saved:',output)
    elif args.command=='import':print(import_phone(args.phone_jsonl,args.output))
    else:print(json.dumps(evaluate(args.trajectory,args.ground_truth,args.align),indent=2))


if __name__=='__main__':main()
