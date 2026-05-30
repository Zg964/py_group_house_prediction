import argparse
import csv
import statistics


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)
    median_price = statistics.median(float(row["house_price"]) for row in train_rows)

    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Id", "house_price"])
        writer.writeheader()
        for row in test_rows:
            writer.writerow({"Id": row["Id"], "house_price": median_price})


if __name__ == "__main__":
    main()
