"""Rasterize VPIN data for livestock tickers (LE, HE)."""

import sys
sys.path.insert(0, r"C:\Users\nicho\PycharmProjects\CTAFlow")

from CTAFlow.features.volume import SequenceRasterizer

def main():
    tickers = ["LE", "HE"]
    rasterizer = SequenceRasterizer(span_pct=0.025)

    for ticker in tickers:
        parquet_path = f"F:\\Upload\\{ticker.lower()}\\{ticker}_1000_1200_vpin.parquet"
        output_path = f"F:\\Upload\\{ticker.lower()}\\{ticker}_rasterized_vpin.npz"

        print(f"\n{'='*50}")
        print(f"Processing {ticker}")
        print(f"{'='*50}")

        try:
            result = rasterizer.parquet_to_npz(
                parquet_path,
                output_path,
                num_bars=4,
                verbose=True
            )
            print(f"Success: {result['num_dates']} dates processed, {result['skipped']} skipped")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
