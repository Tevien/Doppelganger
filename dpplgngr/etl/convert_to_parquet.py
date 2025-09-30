import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv
import argparse

def convert(_input, _output):
    convert_options = pyarrow.csv.ConvertOptions()
    convert_options.column_types = {
        'rate_code': pa.utf8(),
        'store_and_fwd_flag': pa.utf8()
    }

    writer = None
    with pyarrow.csv.open_csv(_input, convert_options=convert_options) as reader:
        for next_chunk in reader:
            if next_chunk is None:
                break
            if writer is None:
                writer = pq.ParquetWriter(_output, next_chunk.schema)
            next_table = pa.Table.from_batches([next_chunk])
            writer.write_table(next_table)
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Parquet")
    parser.add_argument("--input", "-i", help="Input CSV file")
    parser.add_argument("--output", "-o", help="Output Parquet file")
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    convert(input_file, output_file)