import argparse
from src.data.simulate import generate_synthetic_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw/run1.csv')
    parser.add_argument('--minutes', type=int, default=1440)
    args = parser.parse_args()
    df = generate_synthetic_run(length_minutes=args.minutes)
    df.to_csv(args.out, index=False)
    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
